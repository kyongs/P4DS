# math_server.py
from typing import Annotated
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathOperations")

@mcp.tool(
    annotations={
        "title": "Add Numbers",
        "description": (
            "Return the sum of two floating-point numbers.\n\n"
            "Inputs:\n"
            "  • a (float): The first addend. E.g., 3.5\n"
            "  • b (float): The second addend. E.g., 2.1\n\n"
            "Output:\n"
            "  • float: The sum a + b. E.g., 5.6"
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"a": 3.5, "b": 2.1}
        ]
    }
)
def add(
    a: Annotated[float, "First addend (float). E.g., 3.5"],
    b: Annotated[float, "Second addend (float). E.g., 2.1"]
) -> float:
    """Add two numbers and return the result."""
    return a + b


@mcp.tool(
    annotations={
        "title": "Subtract Numbers",
        "description": (
            "Return the difference between two floating-point numbers (a − b).\n\n"
            "Inputs:\n"
            "  • a (float): Minuend. E.g., 5.0\n"
            "  • b (float): Subtrahend. E.g., 2.5\n\n"
            "Output:\n"
            "  • float: The result a − b. E.g., 2.5"
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"a": 5.0, "b": 2.5}
        ]
    }
)
def subtract(
    a: Annotated[float, "Minuend (float). E.g., 5.0"],
    b: Annotated[float, "Subtrahend (float). E.g., 2.5"]
) -> float:
    """Subtract b from a and return the result."""
    return a - b


@mcp.tool(
    annotations={
        "title": "Multiply Numbers",
        "description": (
            "Return the product of two floating-point numbers.\n\n"
            "Inputs:\n"
            "  • a (float): The first factor. E.g., 4.0\n"
            "  • b (float): The second factor. E.g., 2.5\n\n"
            "Output:\n"
            "  • float: The product a × b. E.g., 10.0"
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"a": 4.0, "b": 2.5}
        ]
    }
)
def multiply(
    a: Annotated[float, "First factor (float). E.g., 4.0"],
    b: Annotated[float, "Second factor (float). E.g., 2.5"]
) -> float:
    """Multiply two numbers and return the product."""
    return a * b


@mcp.tool(
    annotations={
        "title": "Divide Numbers",
        "description": (
            "Return the quotient of dividing a by b. Raises if b is zero.\n\n"
            "Inputs:\n"
            "  • a (float): Dividend. E.g., 10.0\n"
            "  • b (float): Divisor; must be nonzero. E.g., 2.0\n\n"
            "Output:\n"
            "  • float: The quotient a ÷ b. E.g., 5.0"
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"a": 10.0, "b": 2.0}
        ]
    }
)
def divide(
    a: Annotated[float, "Dividend (float). E.g., 10.0"],
    b: Annotated[float, "Divisor (float). Must be > 0. E.g., 2.0"]
) -> float:
    """Divide a by b and return the quotient."""
    if b == 0:
        raise ValueError("Divisor (b) cannot be zero.")
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")
