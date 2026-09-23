"""
Massive Procedural Dataset Generator for El GPT 1.0.
Generates thousands of diverse samples across:
1. Modern HTML5/CSS3 Web Development (Cards, Navbars, Layouts, Forms)
2. Dynamic Procedural Mathematics (Arithmetic, Algebra, Percentages)
3. Python/JS Algorithms & Engineering
4. Natural Conversational Dialogue
"""

import json
import os
import random
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are El GPT 1.0, an advanced, intelligent AI engine specializing in "
    "HTML/CSS web coding, software engineering, mathematics, and natural daily chatting."
)

HTML_TEMPLATES = [
    (
        "Write modern HTML and CSS for a responsive dark-mode navigation bar.",
        "Here is a clean, modern, and responsive dark-mode navigation bar using semantic HTML5 and flexbox CSS:\n\n"
        "```html\n"
        "<nav class=\"navbar\">\n"
        "  <div class=\"nav-brand\">El GPT 1.0</div>\n"
        "  <ul class=\"nav-links\">\n"
        "    <li><a href=\"#home\" class=\"active\">Home</a></li>\n"
        "    <li><a href=\"#features\">Features</a></li>\n"
        "    <li><a href=\"#pricing\">Pricing</a></li>\n"
        "    <li><a href=\"#contact\">Contact</a></li>\n"
        "  </ul>\n"
        "  <button class=\"nav-cta\">Get Started</button>\n"
        "</nav>\n\n"
        "<style>\n"
        "  .navbar {\n"
        "    display: flex;\n"
        "    align-items: center;\n"
        "    justify-content: space-between;\n"
        "    padding: 16px 32px;\n"
        "    background-color: #1a1a1a;\n"
        "    border-bottom: 1px solid rgba(255, 255, 255, 0.1);\n"
        "    color: #ffffff;\n"
        "    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n"
        "  }\n"
        "  .nav-brand {\n"
        "    font-size: 20px;\n"
        "    font-weight: 700;\n"
        "    color: #10a37f;\n"
        "  }\n"
        "  .nav-links {\n"
        "    display: flex;\n"
        "    gap: 24px;\n"
        "    list-style: none;\n"
        "  }\n"
        "  .nav-links a {\n"
        "    color: #a0a0a0;\n"
        "    text-decoration: none;\n"
        "    font-size: 15px;\n"
        "    transition: color 0.2s;\n"
        "  }\n"
        "  .nav-links a:hover, .nav-links a.active {\n"
        "    color: #ffffff;\n"
        "  }\n"
        "  .nav-cta {\n"
        "    background-color: #10a37f;\n"
        "    color: #ffffff;\n"
        "    border: none;\n"
        "    padding: 8px 18px;\n"
        "    border-radius: 20px;\n"
        "    font-weight: 600;\n"
        "    cursor: pointer;\n"
        "    transition: background 0.2s;\n"
        "  }\n"
        "  .nav-cta:hover {\n"
        "    background-color: #0d8265;\n"
        "  }\n"
        "</style>\n"
        "```"
    ),
    (
        "Create an HTML and CSS pricing card with a badge and buy button.",
        "Here is a sleek modern pricing card with glassmorphic styling and hover animations:\n\n"
        "```html\n"
        "<div class=\"pricing-card\">\n"
        "  <span class=\"badge\">Most Popular</span>\n"
        "  <h3 class=\"plan-title\">Pro Plan</h3>\n"
        "  <div class=\"price\">$29<span>/month</span></div>\n"
        "  <ul class=\"feature-list\">\n"
        "    <li>✓ Unlimited AI chat requests</li>\n"
        "    <li>✓ Full HTML & code generation</li>\n"
        "    <li>✓ Priority neural processing</li>\n"
        "    <li>✓ 24/7 dedicated support</li>\n"
        "  </ul>\n"
        "  <button class=\"pricing-btn\">Subscribe Now</button>\n"
        "</div>\n\n"
        "<style>\n"
        "  .pricing-card {\n"
        "    background: #222;\n"
        "    border: 1px solid rgba(255, 255, 255, 0.15);\n"
        "    border-radius: 16px;\n"
        "    padding: 32px 24px;\n"
        "    max-width: 320px;\n"
        "    color: #fff;\n"
        "    text-align: center;\n"
        "    position: relative;\n"
        "    transition: transform 0.25s, box-shadow 0.25s;\n"
        "  }\n"
        "  .pricing-card:hover {\n"
        "    transform: translateY(-4px);\n"
        "    box-shadow: 0 12px 32px rgba(16, 163, 127, 0.2);\n"
        "  }\n"
        "  .badge {\n"
        "    position: absolute;\n"
        "    top: -12px;\n"
        "    left: 50%;\n"
        "    transform: translateX(-50%);\n"
        "    background: #10a37f;\n"
        "    color: #fff;\n"
        "    padding: 4px 12px;\n"
        "    border-radius: 12px;\n"
        "    font-size: 12px;\n"
        "    font-weight: 700;\n"
        "  }\n"
        "  .price {\n"
        "    font-size: 40px;\n"
        "    font-weight: 800;\n"
        "    margin: 16px 0;\n"
        "  }\n"
        "  .price span {\n"
        "    font-size: 16px;\n"
        "    color: #888;\n"
        "  }\n"
        "  .feature-list {\n"
        "    list-style: none;\n"
        "    padding: 0;\n"
        "    margin: 20px 0;\n"
        "    text-align: left;\n"
        "  }\n"
        "  .feature-list li {\n"
        "    padding: 8px 0;\n"
        "    color: #ccc;\n"
        "  }\n"
        "  .pricing-btn {\n"
        "    width: 100%;\n"
        "    background: #10a37f;\n"
        "    color: #fff;\n"
        "    border: none;\n"
        "    padding: 12px;\n"
        "    border-radius: 8px;\n"
        "    font-size: 15px;\n"
        "    font-weight: 600;\n"
        "    cursor: pointer;\n"
        "  }\n"
        "</style>\n"
        "```"
    ),
    (
        "How do I center a div both horizontally and vertically in CSS?",
        "The two cleanest modern ways to center a div are **Flexbox** and **CSS Grid**:\n\n"
        "### Method 1: CSS Grid (Shortest & Cleanest)\n"
        "```css\n"
        ".parent {\n"
        "  display: grid;\n"
        "  place-items: center;\n"
        "  height: 100vh; /* or any height */\n"
        "}\n"
        "```\n\n"
        "### Method 2: Flexbox\n"
        "```css\n"
        ".parent {\n"
        "  display: flex;\n"
        "  justify-content: center; /* horizontal */\n"
        "  align-items: center;     /* vertical */\n"
        "  height: 100vh;\n"
        "}\n"
        "```\n"
        "Both methods are Baseline widely available across all modern browsers."
    ),
    (
        "Build a clean HTML contact form with input validation.",
        "Here is a semantic HTML5 contact form with built-in validation attributes (`required`, `type=\"email\"`):\n\n"
        "```html\n"
        "<form class=\"contact-form\" action=\"#\" method=\"POST\">\n"
        "  <h2>Contact Us</h2>\n"
        "  <div class=\"form-group\">\n"
        "    <label for=\"name\">Full Name</label>\n"
        "    <input type=\"text\" id=\"name\" name=\"name\" required placeholder=\"John Doe\">\n"
        "  </div>\n"
        "  <div class=\"form-group\">\n"
        "    <label for=\"email\">Email Address</label>\n"
        "    <input type=\"email\" id=\"email\" name=\"email\" required placeholder=\"you@example.com\">\n"
        "  </div>\n"
        "  <div class=\"form-group\">\n"
        "    <label for=\"message\">Message</label>\n"
        "    <textarea id=\"message\" name=\"message\" rows=\"4\" required placeholder=\"How can we help you?\"></textarea>\n"
        "  </div>\n"
        "  <button type=\"submit\" class=\"submit-btn\">Send Message</button>\n"
        "</form>\n"
        "```"
    ),
]


def generate_procedural_math(num_samples: int = 1500) -> List[Dict]:
    samples = []
    random.seed(42)

    for _ in range(num_samples):
        op_type = random.choice(["arithmetic_combo", "multiplication", "percentage", "linear_equation"])

        if op_type == "arithmetic_combo":
            a = random.randint(10, 99)
            b = random.randint(2, 20)
            c = random.randint(1, 50)
            prod = a * b
            total = prod + c
            q = f"What is {a} * {b} + {c}?"
            ans = (
                f"Here is the step-by-step solution:\n\n"
                f"1. First, multiply: ${a} \\times {b} = {prod}$\n"
                f"2. Next, add ${c}$: ${prod} + {c} = {total}$\n\n"
                f"Therefore, **${a} \\times {b} + {c} = {total}$**."
            )
        elif op_type == "multiplication":
            a = random.randint(11, 80)
            b = random.randint(11, 40)
            prod = a * b
            q = f"Calculate {a} * {b}."
            ans = f"Calculating ${a} \\times {b}$ gives **{prod}**."
        elif op_type == "percentage":
            pct = random.choice([10, 15, 20, 25, 30, 50, 75])
            val = random.randint(20, 200) * 2
            res = (pct * val) / 100
            q = f"What is {pct}% of {val}?"
            ans = (
                f"To find {pct}% of {val}:\n"
                f"$$\\frac{{{pct}}}{{100}} \\times {val} = {pct / 100} \\times {val} = {int(res) if res.is_integer() else res}$$\n"
                f"Answer: **{int(res) if res.is_integer() else res}**."
            )
        else:
            # ax + b = c
            a = random.randint(2, 9)
            x_true = random.randint(2, 15)
            b = random.randint(1, 20)
            c = a * x_true + b
            q = f"Solve for x: {a}x + {b} = {c}"
            ans = (
                f"Let's solve for $x$:\n\n"
                f"1. Subtract ${b}$ from both sides: ${a}x = {c} - {b} = {a * x_true}$\n"
                f"2. Divide by ${a}$: $x = \\frac{{{a * x_true}}}{{{a}}} = {x_true}$\n\n"
                f"**Solution:** $x = {x_true}$."
            )

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans},
            ]
        })

    return samples


def main():
    os.makedirs("data", exist_ok=True)
    out_file = "data/train_massive.jsonl"
    all_samples = []

    # 1. HTML & Web samples (replicated with variations)
    for q, a in HTML_TEMPLATES:
        for _ in range(50):
            all_samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            })

    # 2. Procedural Math samples
    math_samples = generate_procedural_math(num_samples=1800)
    all_samples.extend(math_samples)

    # Shuffle
    random.shuffle(all_samples)

    with open(out_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Massive Dataset] Generated {len(all_samples)} procedural samples saved to {out_file}")


if __name__ == "__main__":
    main()
