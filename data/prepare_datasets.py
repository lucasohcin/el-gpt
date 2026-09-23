"""
Dataset generator for El GPT 1.0.
Compiles a rich multi-domain training dataset focused on:
1. Mathematics (Arithmetic, Algebra, Calculus, Word Problems with LaTeX)
2. Software Coding (Python, JavaScript, Algorithms, Data Structures)
3. Daily Chat & Persona (Friendly dialogue, Persona as El GPT 1.0, Knowledge)
"""

import json
import os
import random
from typing import Dict, List


SYSTEM_PROMPT = "You are El GPT 1.0, an intelligent, helpful, and concise AI engine specializing in mathematics, software coding, and daily conversational chatting."

MATH_PAIRS = [
    (
        "What is 25 * 14 + 50? Show your steps.",
        "Here is the step-by-step calculation:\n\n"
        "1. First, perform the multiplication: $25 \\times 14 = 350$\n"
        "2. Next, add $50$: $350 + 50 = 400$\n\n"
        "Therefore, $25 \\times 14 + 50 = 400$."
    ),
    (
        "How do you solve the quadratic equation $x^2 - 5x + 6 = 0$?",
        "We can solve $x^2 - 5x + 6 = 0$ by factoring or using the quadratic formula.\n\n"
        "**Method 1: Factoring**\n"
        "Find two numbers that multiply to $6$ and add up to $-5$. These numbers are $-2$ and $-3$:\n"
        "$$(x - 2)(x - 3) = 0$$\n\n"
        "Setting each factor to zero:\n"
        "- $x - 2 = 0 \\implies x = 2$\n"
        "- $x - 3 = 0 \\implies x = 3$\n\n"
        "**Solutions:** $x = 2$ and $x = 3$."
    ),
    (
        "What is the derivative of $f(x) = 3x^4 - 5x^2 + 7x - 9$ with respect to $x$?",
        "Using the power rule $\\frac{d}{dx}[x^n] = n x^{n-1}$:\n\n"
        "- $\\frac{d}{dx}[3x^4] = 3 \\cdot 4x^3 = 12x^3$\n"
        "- $\\frac{d}{dx}[-5x^2] = -5 \\cdot 2x = -10x$\n"
        "- $\\frac{d}{dx}[7x] = 7$\n"
        "- $\\frac{d}{dx}[-9] = 0$\n\n"
        "Combining these terms gives:\n"
        "$$f'(x) = 12x^3 - 10x + 7$$"
    ),
    (
        "Evaluate the integral $\\int (2x + 5) \\, dx$.",
        "We integrate term by term using the power rule $\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C$:\n\n"
        "1. $\\int 2x \\, dx = 2 \\cdot \\frac{x^2}{2} = x^2$\n"
        "2. $\\int 5 \\, dx = 5x$\n"
        "3. Add the constant of integration $C$.\n\n"
        "Result:\n"
        "$$\\int (2x + 5) \\, dx = x^2 + 5x + C$$"
    ),
    (
        "A car travels 180 miles in 3 hours. What is its average speed?",
        "The formula for average speed is:\n"
        "$$\\text{Speed} = \\frac{\\text{Distance}}{\\text{Time}}$$\n\n"
        "Given:\n"
        "- Distance = $180$ miles\n"
        "- Time = $3$ hours\n\n"
        "$$\\text{Speed} = \\frac{180}{3} = 60 \\text{ mph}$$\n\n"
        "The car's average speed is **60 miles per hour**."
    ),
    (
        "What is the Pythagorean theorem and how is it used?",
        "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse ($c$) is equal to the sum of the squares of the other two sides ($a$ and $b$):\n\n"
        "$$a^2 + b^2 = c^2$$\n\n"
        "**Example:**\n"
        "If side $a = 3$ and side $b = 4$:\n"
        "$$c^2 = 3^2 + 4^2 = 9 + 16 = 25$$\n"
        "$$c = \\sqrt{25} = 5$$\n\n"
        "It is widely used in geometry, navigation, physics, and computer graphics to calculate Euclidean distance."
    ),
    (
        "Solve for $x$: $3(x - 4) = 2x + 5$",
        "Let's solve step by step:\n\n"
        "1. Distribute the $3$ on the left side:\n"
        "   $$3x - 12 = 2x + 5$$\n"
        "2. Subtract $2x$ from both sides:\n"
        "   $$x - 12 = 5$$\n"
        "3. Add $12$ to both sides:\n"
        "   $$x = 17$$\n\n"
        "**Solution:** $x = 17$."
    ),
    (
        "What is the sum of angles in a triangle?",
        "The interior angles of any Euclidean triangle always sum up to $180^\\circ$ (or $\\pi$ radians)."
    ),
    (
        "If a shirt costs $40 and is on a 25% discount, what is the final price?",
        "To find the final price:\n\n"
        "1. Calculate the discount amount: $40 \\times 0.25 = \\$10$\n"
        "2. Subtract the discount from the original price: $40 - 10 = \\$30$\n\n"
        "Alternatively: $40 \\times (1 - 0.25) = 40 \\times 0.75 = \\$30$.\n"
        "The final price is **$30**."
    ),
    (
        "Explain what Euler's identity is.",
        "Euler's identity is considered one of the most elegant formulas in mathematics:\n\n"
        "$$e^{i\\pi} + 1 = 0$$\n\n"
        "It connects the five most fundamental mathematical constants:\n"
        "- $e$: base of natural logarithms\n"
        "- $i$: the imaginary unit ($\\sqrt{-1}$)\n"
        "- $\\pi$: ratio of a circle's circumference to its diameter\n"
        "- $1$: the multiplicative identity\n"
        "- $0$: the additive identity"
    ),
]

CODE_PAIRS = [
    (
        "Write a Python function to check if a string is a palindrome.",
        "Here is an efficient Python function to check if a string is a palindrome, ignoring case and non-alphanumeric characters:\n\n"
        "```python\n"
        "def is_palindrome(s: str) -> bool:\n"
        "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n\n"
        "# Examples\n"
        "print(is_palindrome('A man, a plan, a canal: Panama')) # True\n"
        "print(is_palindrome('hello'))                         # False\n"
        "```\n\n"
        "**Complexity:** $O(n)$ time and $O(n)$ space."
    ),
    (
        "Write a Python function to calculate the Fibonacci sequence up to n terms.",
        "Here is an efficient generator function to produce the Fibonacci sequence:\n\n"
        "```python\n"
        "def fibonacci(n: int):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        yield a\n"
        "        a, b = b, a + b\n\n"
        "# Example usage\n"
        "print(list(fibonacci(10)))\n"
        "# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
        "```"
    ),
    (
        "How do I implement binary search in Python?",
        "Here is the standard iterative implementation of Binary Search ($O(\\log n)$ time):\n\n"
        "```python\n"
        "def binary_search(arr: list[int], target: int) -> int:\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1  # Target not found\n\n"
        "arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]\n"
        "print(binary_search(arr, 23)) # 5\n"
        "```"
    ),
    (
        "How do you fetch JSON from an API in JavaScript using async/await?",
        "Here is the clean modern way using `fetch` with `async/await` and error handling:\n\n"
        "```javascript\n"
        "async function fetchUserData(url) {\n"
        "  try {\n"
        "    const response = await fetch(url);\n"
        "    if (!response.ok) {\n"
        "      throw new Error(`HTTP error! Status: ${response.status}`);\n"
        "    }\n"
        "    const data = await response.json();\n"
        "    return data;\n"
        "  } catch (error) {\n"
        "    console.error('Failed to fetch data:', error);\n"
        "    throw error;\n"
        "  }\n"
        "}\n"
        "```"
    ),
    (
        "What is the difference between `let`, `const`, and `var` in JavaScript?",
        "Here is the comparison:\n\n"
        "1. `const`:\n"
        "   - Block-scoped.\n"
        "   - Cannot be reassigned.\n"
        "   - Best default choice for variables that don't need re-binding.\n\n"
        "2. `let`:\n"
        "   - Block-scoped.\n"
        "   - Can be reassigned, but cannot be re-declared in the same scope.\n"
        "   - Ideal for loop counters and values that change.\n\n"
        "3. `var`:\n"
        "   - Function-scoped.\n"
        "   - Hoisted to top of scope and can be re-declared.\n"
        "   - Legacy: Avoid in modern ES6+ code."
    ),
    (
        "How do I read and write files in Python safely?",
        "Use Python's `with` statement (context manager) to ensure files are automatically closed:\n\n"
        "```python\n"
        "# Writing to a file\n"
        "with open('output.txt', 'w', encoding='utf-8') as f:\n"
        "    f.write('Hello from El GPT 1.0!\\n')\n\n"
        "# Reading from a file\n"
        "with open('output.txt', 'r', encoding='utf-8') as f:\n"
        "    content = f.read()\n"
        "    print(content)\n"
        "```"
    ),
    (
        "Write a Python script to sort a list of dictionaries by a specific key.",
        "You can use Python's built-in `sorted()` or `list.sort()` with a `key` lambda or `operator.itemgetter`:\n\n"
        "```python\n"
        "users = [\n"
        "    {'name': 'Alice', 'age': 30},\n"
        "    {'name': 'Bob', 'age': 22},\n"
        "    {'name': 'Charlie', 'age': 25},\n"
        "]\n\n"
        "# Sort by age ascending\n"
        "sorted_users = sorted(users, key=lambda u: u['age'])\n"
        "print(sorted_users)\n"
        "# Output: [{'name': 'Bob', 'age': 22}, {'name': 'Charlie', 'age': 25}, {'name': 'Alice', 'age': 30}]\n"
        "```"
    ),
]

CHAT_PAIRS = [
    (
        "Hello! Who are you?",
        "Hello! I am **El GPT 1.0**, your custom AI neural network engine. I am specially trained to help you with mathematics, software coding, and daily conversation. How can I help you today?"
    ),
    (
        "What can you do?",
        "As **El GPT 1.0**, my core capabilities include:\n\n"
        "1. **Mathematics**: Solving arithmetic, algebra, calculus, geometry, and word problems step by step with LaTeX formatting.\n"
        "2. **Coding**: Writing, explaining, and debugging code in Python, JavaScript, HTML/CSS, and other languages.\n"
        "3. **Daily Chat**: Conversing, answering general knowledge questions, brainstorming ideas, and chatting about your day."
    ),
    (
        "How is your day going?",
        "Everything is running smoothly! All neural weights are optimized and ready to assist you. How has your day been going?"
    ),
    (
        "Can you tell me an interesting science fact?",
        "Did you know that a teaspoon of a neutron star would weigh about 6 billion tons on Earth? Neutron stars are so dense that atomic nuclei are packed tightly together, leaving virtually no empty space!"
    ),
    (
        "How does a neural network learn?",
        "A neural network learns through an iterative optimization process:\n\n"
        "1. **Forward Pass**: Input data passes through layers of artificial neurons (weights and biases) to generate a prediction.\n"
        "2. **Loss Calculation**: A loss function measures the error between the model's prediction and the ground truth.\n"
        "3. **Backpropagation**: Calculus (the chain rule) calculates gradients—indicating how much each weight contributed to the error.\n"
        "4. **Optimizer Update**: An algorithm like AdamW updates the weights in the opposite direction of the gradient to reduce error."
    ),
    (
        "Give me three tips to stay productive while coding.",
        "Here are three proven habits for programming productivity:\n\n"
        "1. **Break tasks into atomic sub-tasks**: Never tackle 'build app'; instead tackle 'design database schema', 'create one route', etc.\n"
        "2. **Use the 25-minute Pomodoro focus block**: 25 minutes of deep distraction-free flow followed by a 5-minute break.\n"
        "3. **Write tests and commit frequently**: Small, tested commits eliminate debugging fatigue and keep your mental cache clear."
    ),
    (
        "Good morning!",
        "Good morning! Ready for a productive day? Whether you want to tackle math equations, write code, or just chat, I'm here for you."
    ),
    (
        "Thank you for the help!",
        "You're very welcome! If you have any more math problems, coding questions, or anything else you'd like to explore, feel free to ask anytime."
    ),
]


def generate_all_samples() -> List[Dict]:
    dataset = []

    # Combine pairs
    all_pairs = []
    # Add multiple variations / replicates for robust learning
    for p in MATH_PAIRS:
        all_pairs.append(p)
    for p in CODE_PAIRS:
        all_pairs.append(p)
    for p in CHAT_PAIRS:
        all_pairs.append(p)

    # Replicate pairs with minor variations / order shuffling
    augmented_pairs = all_pairs * 8
    random.seed(42)
    random.shuffle(augmented_pairs)

    for user_msg, asst_msg in augmented_pairs:
        dataset.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg},
            ]
        })

    return dataset


def main():
    os.makedirs("data", exist_ok=True)
    samples = generate_all_samples()

    # 90% train, 10% val split
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_path = "data/train.jsonl"
    val_path = "data/val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Dataset Generator] Created {len(train_samples)} training samples -> {train_path}")
    print(f"[Dataset Generator] Created {len(val_samples)} validation samples -> {val_path}")


if __name__ == "__main__":
    main()
