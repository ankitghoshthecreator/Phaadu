Phaadu the programming language

A skeleton

Source text
   │
   ▼
Lexer  →  tokens  →  Parser  →  AST  →  Semantic checks
   │
   ▼
Intermediate Representation (IR)
   │
   ▼
LLVM optimizer
   │
   ▼
Assembly
   │
   ▼
Machine code  (.exe)

 What’s Really Happening Here


| Stage             | Real-World Meaning             | Rust line that imitates it  |
| ----------------- | ------------------------------ | --------------------------- |
| **Lexer**         | Turns text into tokens         | `source.split_whitespace()` |
| **Parser**        | Builds structure of operations | printed “Parsed as …”       |
| **IR → Assembly** | Translates to CPU operations   | printed assembly lines      |



Why We Need Each Step (and what breaks if skipped)


| Step         | Purpose                        | If you skip it                            |
| Lexer        | Separates words/symbols        | Compiler can’t tell what’s code vs space  |
| Parser       | Finds structure (who adds who) | No order → wrong math (5 + 3 * 2 problem) |
| IR / Codegen | Converts logic to CPU ops      | CPU can’t run text                        |
| Assembly     | Actual CPU instructions        | Program never runs                        |



Where LLVM Fits
LLVM (Low-Level Virtual Machine)

A framework that handles the IR → assembly → machine code part for you.

Instead of writing your own assembler, you hand LLVM your program in its internal format (LLVM IR), and LLVM emits optimized native code.

Rust itself uses LLVM.
Later, your language compiler (written in Rust) will use LLVM too, through crates like inkwell or llvm-sys.



