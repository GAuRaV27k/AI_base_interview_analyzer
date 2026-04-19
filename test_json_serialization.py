#!/usr/bin/env python3
"""
Test script to verify JSON serialization of analysis results.
Validates that all data types can be properly JSON-encoded.
"""

import json
import sys
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from api.utils.responses import _convert_to_serializable


def test_numpy_conversion():
    """Test that numpy types are converted to JSON-serializable types."""
    print("Testing numpy type conversion...")
    
    test_cases = {
        "numpy_int64": np.int64(42),
        "numpy_float64": np.float64(3.14159),
        "numpy_array": np.array([1, 2, 3, 4, 5]),
        "numpy_2d_array": np.array([[1.1, 2.2], [3.3, 4.4]]),
        "dict_with_numpy": {
            "value": np.float64(10.5),
            "array": np.array([1, 2, 3])
        },
        "list_with_numpy": [np.int64(1), np.float64(2.5), "text"],
    }
    
    for name, value in test_cases.items():
        try:
            converted = _convert_to_serializable(value)
            # Verify it's JSON-serializable
            json_str = json.dumps(converted)
            print(f"  ✓ {name}: {type(value).__name__} → {type(converted).__name__}")
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
            return False
    
    return True


def test_emotion_breakdown():
    """Test realistic emotion breakdown dict."""
    print("Testing emotion breakdown serialization...")
    
    emotion_breakdown = {
        "Enthusiastic": np.float64(35.5),
        "Calm": np.float64(25.2),
        "Neutral": np.float64(20.1),
        "Nervous": np.float64(10.2),
        "Surprised": np.float64(9.0),
    }
    
    try:
        converted = _convert_to_serializable(emotion_breakdown)
        json_str = json.dumps(converted)
        data = json.loads(json_str)
        print(f"  ✓ emotion_breakdown: {len(data)} emotions")
        return True
    except Exception as e:
        print(f"  ✗ emotion_breakdown: FAILED - {e}")
        return False


def test_emotion_timeline():
    """Test realistic emotion timeline (list of int predictions)."""
    print("Testing emotion timeline serialization...")
    
    # Simulate emotion timeline: per-frame predictions
    emotion_timeline = [np.int64(i % 7) for i in range(150)]
    
    try:
        converted = _convert_to_serializable(emotion_timeline)
        json_str = json.dumps(converted)
        data = json.loads(json_str)
        print(f"  ✓ emotion_timeline: {len(data)} frames")
        return True
    except Exception as e:
        print(f"  ✗ emotion_timeline: FAILED - {e}")
        return False


def test_full_analysis_result():
    """Test full analysis result dict."""
    print("Testing full analysis result...")
    
    result = {
        "confidence_score": np.float64(85.5),
        "speech_rate": np.float64(145.3),
        "eye_contact_score": np.float64(78.9),
        "emotion_prediction": "Enthusiastic",
        "final_interview_score": np.float64(82.1),
        "transcript": "This is a test transcript.",
        "word_count": np.int64(42),
        "emotion_breakdown": {
            "Enthusiastic": np.float64(35.5),
            "Calm": np.float64(25.2),
            "Neutral": np.float64(20.1),
            "Nervous": np.float64(10.2),
            "Surprised": np.float64(9.0),
        },
        "emotion_timeline": [np.int64(i % 7) for i in range(150)],
        "frames_analyzed": np.int64(300),
        "faces_detected": np.int64(285),
        "audio_energy": np.float64(0.12345),
        "audio_duration": np.float64(45.5),
        "audio_error": "",
    }
    
    try:
        converted = _convert_to_serializable(result)
        json_str = json.dumps(converted)
        data = json.loads(json_str)
        print(f"  ✓ Full result: {len(data)} fields")
        print(f"    - confidence_score: {data['confidence_score']}")
        print(f"    - emotion_breakdown keys: {list(data['emotion_breakdown'].keys())}")
        print(f"    - emotion_timeline length: {len(data['emotion_timeline'])}")
        return True
    except Exception as e:
        print(f"  ✗ Full result: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("JSON Serialization Test Suite")
    print("=" * 70)
    
    all_pass = True
    all_pass &= test_numpy_conversion()
    print()
    all_pass &= test_emotion_breakdown()
    print()
    all_pass &= test_emotion_timeline()
    print()
    all_pass &= test_full_analysis_result()
    
    print()
    print("=" * 70)
    if all_pass:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
