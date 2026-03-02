import re
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_lightweight_stubs() -> None:
    """Allow importing project modules in envs without torch/transformers."""
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _NoGrad:
            def __call__(self, fn=None):
                if fn is None:
                    return self
                return fn

            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch_mod.no_grad = lambda: _NoGrad()
        torch_mod.manual_seed = lambda seed: None
        torch_mod.float16 = "float16"
        torch_mod.bfloat16 = "bfloat16"
        torch_mod.float32 = "float32"
        torch_mod.Tensor = object
        torch_mod.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda seed: None,
        )

        nn_mod = types.ModuleType("torch.nn")
        functional_mod = types.ModuleType("torch.nn.functional")
        functional_mod.normalize = lambda x, p=2, dim=1: x
        nn_mod.functional = functional_mod

        sys.modules["torch"] = torch_mod
        sys.modules["torch.nn"] = nn_mod
        sys.modules["torch.nn.functional"] = functional_mod

    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")

        class _DummyModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

        class _DummyTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        transformers_mod.AutoModel = _DummyModel
        transformers_mod.AutoTokenizer = _DummyTokenizer
        transformers_mod.AutoModelForCausalLM = _DummyModel
        sys.modules["transformers"] = transformers_mod


_install_lightweight_stubs()


from timeqa_baseline_lab.chunking import Chunk
from timeqa_baseline_lab.config import load_config
from timeqa_baseline_lab.data import load_corpus, load_questions_from_arrow
from timeqa_baseline_lab.strategies import rag_cot, react, zero_shot_cot


class InMemoryFullCorpusRetriever:
    """Lightweight retriever for tests using full corpus as retrieval source."""

    def __init__(self, docs):
        self.chunks = [
            Chunk(
                chunk_id=f"{d.doc_id}-chunk-0000",
                doc_id=d.doc_id,
                title=d.title,
                source_idx=d.source_idx,
                text=d.content,
                start_token=0,
                end_token=max(1, len(d.content.split())),
            )
            for d in docs
        ]

    @staticmethod
    def _terms(text: str):
        return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]

    def search_with_scores(self, query: str, top_k: int = 5):
        q_terms = self._terms(query)
        scored = []
        for chunk in self.chunks:
            hay = f"{chunk.title} {chunk.text}".lower()
            score = sum(1.0 for t in q_terms if t in hay)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: min(top_k, len(scored))]

    def search(self, query: str, top_k: int = 5):
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]


class StubLLM:
    """Deterministic LLM stub to unit-test strategy control flow."""

    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, prompt: str, system_prompt=None) -> str:  # noqa: ARG002
        text = prompt.strip()

        # QAaP parse
        if "Question parsing:" in text and "defines `query` and `answer_key`" in text:
            return (
                "```python\n"
                "query = {\"subject\": None, \"relation\": \"related to\", \"object\": None, "
                "\"time\": {\"start\": None, \"end\": None}}\n"
                "answer_key = \"object\"\n"
                "```"
            )

        # QAaP search program
        if "Search:" in text and "defines `entities_to_search`" in text:
            return "```python\nentities_to_search = [\"test entity\"]\n```"

        # QAaP extract information blocks
        if "Extract information relevant to the query" in text and "information.append" in text:
            return (
                "```python\n"
                "information.append({\"subject\": \"test subject\", \"relation\": \"related to\", "
                f"\"object\": \"{self.answer}\", "
                "\"time\": {\"start\": datetime(1985, 1, 1), \"end\": datetime(1989, 12, 31)}})\n"
                "```"
            )

        # ReAct iterative steps
        if text.endswith("Thought 1:"):
            return "Find evidence first.\nAction 1: Search[test entity]"
        if text.endswith("Thought 2:"):
            return f"Enough evidence.\nAction 2: Finish[{self.answer}]"
        if text.endswith("Thought 3:"):
            return f"Finish now.\nAction 3: Finish[{self.answer}]"

        # MRAG keyword extraction
        if "<Keywords>" in text:
            return "[\"test\", \"entity\", \"time\"]"

        # MRAG QFS prompt
        if "<Summarization>" in text:
            return (
                "<Summarization>\n"
                f"The evidence indicates the answer is {self.answer}.\n"
                "</Summarization>"
            )

        # MRAG combiner prompt
        if "<Answer>" in text and "Now your context paragraph and question are" in text:
            return f"<Answer>\n{self.answer}\n</Answer>"

        # Generic fallback used by zero-shot / rag / other final answers
        return f"Final Answer: {self.answer}"


class TestSingleQuestionStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        cls.docs = load_corpus(cfg.data.corpus_path)
        cls.question_item = load_questions_from_arrow(cfg.data.question_arrow_path, limit=1)[0]
        cls.top_k = cfg.retriever.top_k

        cls.expected_answer = cls.question_item.targets[0] if cls.question_item.targets else "unknown"
        cls.retriever = InMemoryFullCorpusRetriever(cls.docs)

    def _make_llm(self):
        return StubLLM(str(self.expected_answer))

    def test_full_corpus_is_loaded_for_retrieval(self):
        self.assertEqual(len(self.retriever.chunks), len(self.docs))
        self.assertGreater(len(self.retriever.chunks), 700)

    def test_zero_shot_cot_single_question(self):
        out = zero_shot_cot(self._make_llm(), self.question_item.question)
        self.assertTrue(out.answer)
        self.assertEqual(len(out.retrieved), 0)

    def test_rag_cot_single_question_with_full_corpus_retrieval(self):
        out = rag_cot(self._make_llm(), self.retriever, self.question_item.question, self.top_k)
        self.assertTrue(out.answer)
        self.assertGreaterEqual(len(out.retrieved), 1)
        self.assertLessEqual(len(out.retrieved), self.top_k)

    def test_react_single_question_with_full_corpus_retrieval(self):
        out = react(self._make_llm(), self.retriever, self.question_item.question, self.top_k)
        self.assertTrue(out.answer)
        self.assertGreaterEqual(len(out.retrieved), 1)
        self.assertGreaterEqual(len(out.trace), 1)



if __name__ == "__main__":
    unittest.main()