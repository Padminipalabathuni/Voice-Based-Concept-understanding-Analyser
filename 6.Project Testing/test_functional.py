import sys
sys.path.append('../5. Project Development Phase')

from speech_to_text import filler_word_ratio
from scoring_engine import evaluate_understanding, calculate_fluency_score
from semantic_eval import semantic_similarity, get_reference

def test_filler_word_ratio():
    text = "um this is like basically a test"
    ratio = filler_word_ratio(text)
    assert ratio > 0, "Filler ratio should be greater than 0"
    print(f"✅ Filler ratio test passed: {ratio}")

def test_scoring_engine():
    audio = {"pause_ratio": 0.2, "rms_energy": 0.05}
    score, level, color = evaluate_understanding(0.8, 0.02, audio)
    assert score >= 80, "High similarity should give high score"
    assert level == "Strong Understanding"
    print(f"✅ Scoring test passed: {score}/100 - {level}")

def test_poor_understanding():
    audio = {"pause_ratio": 0.5, "rms_energy": 0.005}
    score, level, color = evaluate_understanding(0.1, 0.2, audio)
    assert level == "Poor Understanding"
    print(f"✅ Poor understanding test passed: {score}/100")

def test_semantic_similarity():
    text1 = "machine learning is a subset of AI"
    text2 = "machine learning enables systems to learn from data"
    score = semantic_similarity(text1, text2)
    assert 0 <= score <= 1, "Similarity should be between 0 and 1"
    print(f"✅ Semantic similarity test passed: {score}")

def test_fluency_score():
    result = calculate_fluency_score(0.02, 0.1)
    assert result["fluency_score"] >= 0
    print(f"✅ Fluency test passed: {result}")

def test_reference_exists():
    ref = get_reference("Machine Learning")
    assert len(ref) > 0
    print(f"✅ Reference test passed!")

if __name__ == "__main__":
    print("Running functional tests...\n")
    test_filler_word_ratio()
    test_scoring_engine()
    test_poor_understanding()
    test_semantic_similarity()
    test_fluency_score()
    test_reference_exists()
    print("\n✅ All functional tests passed!")
