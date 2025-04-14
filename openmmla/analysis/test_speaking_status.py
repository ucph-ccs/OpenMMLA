from openmmla.analysis.measurements import speaking_status


def test_speaking_status():
    # Test Case 1: Silent case
    speaker_recognition_1 = [
        {
            'speakers': '["silent"]',
            'time': 1734094853.0
        }
    ]
    badge_relations_1 = []
    status1, speaker_statuses1 = speaking_status(speaker_recognition_1, badge_relations_1)
    assert status1 == 0
    assert speaker_statuses1 == {}

    # Test Case 2: Single speaker, not facing anyone
    speaker_recognition_2 = [
        {
            'speakers': '[1]',
            'time': 1734094853.0
        }
    ]
    badge_relations_2 = [
        {
            'graph': '{"1": []}',
            'time': 1734094853.5
        }
    ]
    status2, speaker_statuses2 = speaking_status(speaker_recognition_2, badge_relations_2)
    assert status2 == 1
    assert speaker_statuses2 == {1: [(1, 1734094853.0)]}

    # Test Case 3: Multiple speakers, one facing others
    speaker_recognition_3 = [
        {
            'speakers': '[1, 2]',
            'time': 1734094853.0
        }
    ]
    badge_relations_3 = [
        {
            'graph': '{"1": ["2"], "2": []}',
            'time': 1734094853.5
        }
    ]
    status3, speaker_statuses3 = speaking_status(speaker_recognition_3, badge_relations_3)
    assert status3 == 2
    assert 1 in speaker_statuses3 and 2 in speaker_statuses3
    assert speaker_statuses3[1][0][0] == 2  # Speaker 1 should have status 2

    # Test Case 4: Multiple time points for same speaker
    speaker_recognition_4 = [
        {
            'speakers': '[1]',
            'time': 1734094853.0
        },
        {
            'speakers': '[1]',
            'time': 1734094856.0
        }
    ]
    badge_relations_4 = [
        {
            'graph': '{"1": ["2"]}',
            'time': 1734094853.5
        }
    ]
    status4, speaker_statuses4 = speaking_status(speaker_recognition_4, badge_relations_4)
    assert status4 == 2
    assert len(speaker_statuses4[1]) == 2  # Should have two time points

    # Test Case 5: Time threshold test
    speaker_recognition_5 = [
        {
            'speakers': '[1]',
            'time': 1734094853.0
        }
    ]
    badge_relations_5 = [
        {
            'graph': '{"1": ["2"]}',
            'time': 1734094856.0  # 3 seconds later, should not affect status
        }
    ]
    status5, speaker_statuses5 = speaking_status(speaker_recognition_5, badge_relations_5)
    assert status5 == 1  # Should remain 1 as facing event is too far in time

    # Test Case 6: Complex scenario with multiple speakers and times
    speaker_recognition_6 = [
        {
            'speakers': '[1, 2, "silent"]',
            'time': 1734094853.0
        },
        {
            'speakers': '[1, 3]',
            'time': 1734094855.0
        },
        {
            'speakers': '[2, 3]',
            'time': 1734094857.0
        }
    ]
    badge_relations_6 = [
        {
            'graph': '{"1": ["2"], "2": ["3"], "3": []}',
            'time': 1734094853.5
        },
        {
            'graph': '{"1": [], "2": [], "3": ["1"]}',
            'time': 1734094856.0
        }
    ]
    status6, speaker_statuses6 = speaking_status(speaker_recognition_6, badge_relations_6)
    assert status6 == 2
    assert len(speaker_statuses6) == 3  # Should have records for 3 speakers

    # Test Case 7: Invalid data handling
    speaker_recognition_7 = [
        {
            'speakers': 'invalid_json',
            'time': 1734094853.0
        }
    ]
    badge_relations_7 = [
        {
            'graph': 'invalid_json',
            'time': 1734094853.5
        }
    ]
    status7, speaker_statuses7 = speaking_status(speaker_recognition_7, badge_relations_7)
    assert status7 == 0
    assert speaker_statuses7 == {}


if __name__ == "__main__":
    # Run all test cases
    test_speaking_status()
    print("All tests passed!")

    # Individual example for manual testing
    speaker_recognition = [
        {
            'speakers': '[0, "silent"]',
            'time': 1734094853.0
        },
        {
            'speakers': '[0, 1]',
            'time': 1734094855.0
        },
        {
            'speakers': '[1, 2]',
            'time': 1734094857.0
        }
    ]

    badge_relations = [
        {
            'graph': '{"0": ["1"], "1": ["2"], "2": []}',
            'time': 1734094855.5
        },
        {
            'graph': '{"0": [], "1": ["0"], "2": ["1"]}',
            'time': 1734094857.5
        }
    ]

    status, speaker_statuses = speaking_status(speaker_recognition, badge_relations)
    print(f"Final status: {status}")
    print("Speaker statuses:")
    for speaker, records in speaker_statuses.items():
        print(f"Speaker {speaker}: {records}")
