class RingBuffer:
    """Circular buffer for storing data streams."""
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.full = False

    def push(self, item):
        if self.full:
            self.head = (self.head + 1) % self.size
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.size
        self.full = self.tail == self.head

    def get(self):
        if not self.full and self.head == self.tail:
            return []
        if self.head < self.tail:
            return self.buffer[self.head:self.tail]
        return self.buffer[self.head:] + self.buffer[:self.tail]

    def __len__(self):
        if self.full:
            return self.size
        if self.tail >= self.head:
            return self.tail - self.head
        return self.size - (self.head - self.tail)


# Example usage:
buffer = RingBuffer(5)
for i in range(7):
    buffer.push(i)
    print(f"Buffer after pushing {i}: {buffer.get()}")

print(f"Final buffer: {buffer.get()}")
print(f"Buffer length: {len(buffer)}")
