class A:
    def __init__(self, type):
        self.type = type

        # Map the 'type' to the corresponding record function during initialization
        self.record_func = {
            1: self.record_1,
            2: self.record_2,
            3: self.record_3
        }.get(self.type, self.default_record)  # Fallback to a default method if type is invalid

    def record(self):
        # Call the corresponding record function without needing to check the type every time
        return self.record_func()

    def record_1(self):
        return "Recording type 1"

    def record_2(self):
        return "Recording type 2"

    def record_3(self):
        return "Recording type 3"

    def default_record(self):
        return "Unknown record type"


# Usage:
a1 = A(1)
print(a1.record())  # Output: "Recording type 1"

a2 = A(2)
print(a2.record())  # Output: "Recording type 2"

a3 = A(3)
print(a3.record())  # Output: "Recording type 3"

a_invalid = A(99)
print(a_invalid.record())  # Output: "Unknown record type"
