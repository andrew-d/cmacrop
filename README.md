# cmacrop

[![Build Status](https://travis-ci.org/andrew-d/cmacrop.svg?branch=master)](https://travis-ci.org/andrew-d/cmacrop)

## What Is This?

cmacrop is an enhanced macro system for C.  This project was heavily inspired
by [cmacro](https://github.com/eudoxia0/cmacro), which implements "Lisp Macros
For C", and is written in Common Lisp.  cmacrop is a very similar
implementation in Python.

## FAQ

- *Q: Why did you rewrite this in Python?*
- A: I wanted to use something that didn't require a bunch of tooling, frankly.
  The original project is far more elegant, and currently has more features
  implemented, but it requires one to compile two binaries (the lexer and the
  macro processor), requiring [SBCL](http://en.wikipedia.org/wiki/Steel_Bank_Common_Lisp)
  and a C compiler.  I wanted a single file that I could drop into an existing
  project without requiring the Lisp toolchain.

- *Q: Why would you use this?*
- A: For answers to this, I refer you to the original [cmacro readme](https://github.com/eudoxia0/cmacro),
  which is an excellent introduction to what macros are and why they're useful.

## Usage

- Download `cmacro.py`, either by cloning the repository or using
  [this link](https://github.com/andrew-d/cmacrop/raw/master/cmacro.py).
- Ensure you have Python 2.7+ installed on your computer
- Run it:
      python cmacro.py [command] file.c
  For example:
      python cmacro.py expand file_with_macros.c > expanded.c

For more usage information, refer to the help: `python cmacro.py --help`

## Status

This tool is currently under development, but should be functional.  I'm in the
process of adding more tests and expanding some of the functionality (e.g. the
support for commands), but the basics should work fine.

### Things Remaining To Do

- Add macro commands
- Add flags to control preprocessing / output
- Add support for customization:
  - Custom commands
  - External commands to generate templates/toplevels - are passed a JSON AST
    to STDIN and should print a new one to STDOUT.
