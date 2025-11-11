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


Your source code   →   Lexer   →   Parser   →   AST   →   IR
                                                           |
                                                           v
                                                          LLVM
                                                           |
                                                           v 
                                                 Machine Code (.exe)

# What would make your program produce a real .exe

## You would have to:

Generate LLVM IR (a low-level intermediate code)

Pass it to the LLVM backend

Tell LLVM to “emit” (write) the machine code to disk

Link all system libraries

Save the result, e.g. program.exe (Windows) or program.out (Linux)

Then you could double-click that .exe and run it — no Rust or Cargo needed.


# Summary Diagram
Step 1: You write Rust code for your compiler
↓
Cargo → rustc → LLVM
↓
YourCompiler.exe
↓
Step 2: Users write programs in your language
↓
YourCompiler.exe reads those files, builds AST, calls LLVM
↓
LLVM emits → program.exe  (the user’s compiled program)


# NOTE
Cargo uses Rust’s compiler (which already includes LLVM) to build my compiler as an .exe. 
But my compiler still needs to use LLVM inside Rust to generate .exe files for programs written 
in my new language