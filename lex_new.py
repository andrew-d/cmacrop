#!/usr/bin/env python

from __future__ import print_function

import sys
import string
from collections import namedtuple


TOK_IDENTIFIER      = 'TOK_IDENTIFIER'
TOK_I_CONSTANT      = 'TOK_I_CONSTANT'
TOK_F_CONSTANT      = 'TOK_F_CONSTANT'
TOK_STRING_LITERAL  = 'TOK_STRING_LITERAL'
TOK_OPERATOR        = 'TOK_OPERATOR'


# A token has
#   type:  one of the TOK_* constants
#   value: string value, as taken from input
#   line:  line in the input
#   col:   column in the input
#
Token = namedtuple('Token', 'type value line col')


class LexerError(Exception):
    pass


class CLexer(object):
    """
    A very basic lexer for C.  Initialize with the input string, and
    then call lex(), which generates tokens. None is generated at EOF
    (and the generator expires).
    """
    def __init__(self, input):
        self.input = input
        self.pos = 0
        self.curstart = 0

        self.state = self._lex_general
        self.line = 1

    def lex(self):
        while self.state:
            self.state = yield from self.state()

    #--------- Internal ---------#

    def _eof(self):
        return self.pos >= len(self.input)

    def _emit(self, toktype):
        # Find the column by looking back to the previous newline.
        newline_pos = self.input.rfind('\n', 0, self.curstart)
        if newline_pos == -1:
            newline_pos = 0

        # We add 1 so that the first column isn't index 0
        col = self.curstart - newline_pos + 1

        tok = Token(toktype,
                    self.input[self.curstart:self.pos],
                    self.line,
                    col)
        self.curstart = self.pos
        return tok

    WHITESPACE = set([' ', '\t', '\v', '\n', '\f'])

    def _lex_general(self):
        # Read until we hit a non-whitespace character.
        while not self._eof() and self.input[self.pos] in self.WHITESPACE:
            if self.input[self.pos] == '\n':
                self.line += 1
            self.pos += 1

        if self._eof():
            yield None
            return None
        else:
            # Skip emitting whitespace.
            self.curstart = self.pos

        # Depending on the current character, either lex an identifier, a
        # number, a string literal, or an operator.
        ch = self.input[self.pos]
        if ch in string.ascii_letters or ch == '_':
            return self._lex_identifier
        elif ch in string.digits:
            return self._lex_number
        elif ch == '"':
            return self._lex_string
        else:
            return self._lex_operator

    IDENTIFIERS = set(string.ascii_letters + string.digits + "_")
    def _lex_identifier(self):
        while not self._eof() and self.input[self.pos] in self.IDENTIFIERS:
            self.pos += 1

        if self._eof():
            yield None
            return None

        yield self._emit(TOK_IDENTIFIER)
        return self._lex_general

    DIGITS = set(string.digits)
    def _lex_number(self):
        while not self._eof() and self.input[self.pos] in self.DIGITS:
            self.pos += 1

        if self._eof():
            yield None
            return None

        yield self._emit(TOK_I_CONSTANT)
        return self._lex_general

    def _lex_string(self):
        while not self._eof():
            self.pos += 1

            if self.input[self.pos] == '"':
                self.pos += 1
                yield self._emit(TOK_STRING_LITERAL)
                return self._lex_general

            # If this is a string escape, we skip the next character.
            # TODO: escape end-of-line?
            if self.input[self.pos] == '\\':
                self.pos += 1

            # TODO: hex escape

        # If we reach here, EOF.
        yield None
        return None

    SEPARATORS = set(',;()[]{}.')
    SYMBOL_SET = set(string.printable) - SEPARATORS - IDENTIFIERS
    def _lex_operator(self):
        if self.input[self.pos:self.pos+3] == '...':
            self.pos += 3
            yield self._emit(TOK_OPERATOR)
            return self._lex_general
        elif self.input[self.pos] in self.SEPARATORS:
            self.pos += 1
            yield self._emit(TOK_OPERATOR)
            return self._lex_general

        while not self._eof() and self.input[self.pos] in self.SYMBOL_SET:
            self.pos += 1

        if self._eof():
            yield None
            return None

        yield self._emit(TOK_I_CONSTANT)
        return self._lex_general


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        text = f.read()

    lexer = CLexer(text)
    for t in lexer.lex():
        print(t)
