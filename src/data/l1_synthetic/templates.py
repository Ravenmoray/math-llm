"""Natural-language framings for arithmetic problems. Variety prevents template-overfitting."""

ADD_PROMPTS = [
    "What is {a} + {b}?",
    "Compute {a} + {b}.",
    "Find the sum of {a} and {b}.",
    "Add {a} and {b}.",
    "If you start with {a} and add {b}, what do you get?",
    "{a} plus {b} equals what?",
    "Calculate {a} + {b}.",
]

SUB_PROMPTS = [
    "What is {a} - {b}?",
    "Compute {a} - {b}.",
    "Find the difference {a} - {b}.",
    "Subtract {b} from {a}.",
    "If you have {a} and take away {b}, how much is left?",
    "{a} minus {b} equals what?",
]

MUL_PROMPTS = [
    "What is {a} × {b}?",
    "Compute {a} × {b}.",
    "Find the product of {a} and {b}.",
    "Multiply {a} by {b}.",
    "{a} times {b} equals what?",
    "Calculate the product {a} · {b}.",
]

DIV_PROMPTS = [
    "What is {a} ÷ {b}?",
    "Compute {a} ÷ {b}.",
    "Divide {a} by {b}.",
    "Find the quotient when {a} is divided by {b}.",
    "How many times does {b} go into {a}?",
]

FRAC_ADD_PROMPTS = [
    "What is {a}/{b} + {c}/{d}?",
    "Add the fractions {a}/{b} and {c}/{d}.",
    "Compute {a}/{b} + {c}/{d} and simplify.",
    "Find the sum of {a}/{b} and {c}/{d}.",
]

FRAC_MUL_PROMPTS = [
    "What is ({a}/{b}) × ({c}/{d})?",
    "Multiply {a}/{b} by {c}/{d}.",
    "Compute the product ({a}/{b}) · ({c}/{d}) and simplify.",
]

PERCENT_OF_PROMPTS = [
    "What is {p}% of {n}?",
    "Find {p}% of {n}.",
    "Compute {p} percent of {n}.",
    "If something costs ${n}, what is {p}% of that amount?",
]

ORDER_OPS_PROMPTS = [
    "Evaluate {expr}.",
    "Compute {expr} using the order of operations.",
    "What is the value of {expr}?",
    "Simplify {expr}.",
]

WORD_ADD_TEMPLATES = [
    "{name} has {a} {item}. {name2} gives them {b} more {item}. How many {item} does {name} have now?",
    "A box contains {a} {item}. Another box contains {b} {item}. How many {item} are there in total?",
    "{name} collected {a} {item} on Monday and {b} {item} on Tuesday. How many {item} did {name} collect in all?",
]

WORD_SUB_TEMPLATES = [
    "{name} has {a} {item}. {name} gives {b} to {name2}. How many {item} does {name} have left?",
    "There were {a} {item} in the basket. {b} were taken out. How many remain?",
    "{name} had {a} {item} and used {b}. How many are left?",
]

WORD_MUL_TEMPLATES = [
    "{name} buys {a} boxes of {item}. Each box contains {b} {item}. How many {item} does {name} have in total?",
    "A bag holds {b} {item}. How many {item} are in {a} bags?",
    "{a} {item} are sold at ${b} each. What is the total cost?",
]

WORD_DIV_TEMPLATES = [
    "{name} has {a} {item} and wants to share them equally among {b} friends. How many does each friend get?",
    "A teacher divides {a} {item} evenly among {b} students. How many does each student receive?",
    "{a} {item} are packed into groups of {b}. How many groups can be made?",
]

NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
         "Avery", "Quinn", "Drew", "Reese", "Skylar", "Cameron", "Dakota"]

ITEMS = ["apples", "books", "marbles", "coins", "stickers", "pencils", "cards",
         "cookies", "toys", "stamps", "candies", "beads", "oranges", "erasers"]
