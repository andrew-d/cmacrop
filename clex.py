#!/usr/bin/env python

import sys

from ply import lex

# Thanks to cmacro:
#  https://github.com/eudoxia0/cmacro/blob/master/grammar/lexing.l
# for much of the inspiration and lexing concepts

class CLexer(object):

    octal_digits = "[0-7]"
    decimal_digits = "[0-9]"
    positive_decimal_digits = "[1-9]"
    letters_and_underscores = "[a-zA-Z_]"
    alphanumeric = "[a-zA-Z_0-9]"
    hexadecimal = "[a-fA-F0-9]"
    hexadecimal_prefix = "(0[xX])"
    exponent_part = "([Ee][+-]?" + decimal_digits + "+)"
    binary_exponent = "([Pp][+-]?" + decimal_digits + "+)"
    float_suffixes = "(f|F|l|L)"
    int_suffixes = "(((u|U)(l|L|ll|LL)?)|((l|L|ll|LL)(u|U)?))"
    char_prefixes = "(u|U|L)"
    string_prefixes = "(u8|u|U|L)"

    # TODO: this should parse anything
    escape_sequence = r"""(\\(['"\?\\abfnrtv]|[0-7]{1,3}|x[a-fA-F0-9]+))"""

    whitespace = r"[ \t\v\n\f]"
    separator = r"[\,\;\(\)\[\]\{\}\.]"
    symbol = r"[^ \t\v\n\f\,\;\(\)\[\]\{\}\.a-zA-Z_0-9]"

    tokens = [
        'IDENTIFIER',
        'I_CONSTANT',
        'F_CONSTANT',
        'STRING_LITERAL',
        'OPERATOR',
        'WHITESPACE',
    ]

    t_IDENTIFIER = letters_and_underscores + alphanumeric + "+"

    t_I_CONSTANT = (
        "(" +
        hexadecimal_prefix + hexadecimal + "+" + int_suffixes + "?|" +
        positive_decimal_digits + decimal_digits + "*" + int_suffixes + "?|" +
        "0" + octal_digits + "+" + int_suffixes + "?|" +
        char_prefixes + r"?\'([^'\\\n]|" + escape_sequence + r")+\'"
        ")"
    )

    t_F_CONSTANT = (
        "(" +
        decimal_digits + "+" + exponent_part + float_suffixes + "?|" +
        decimal_digits + r"*\." + decimal_digits + "+" + exponent_part + "?" + float_suffixes + "?|" +
        hexadecimal_prefix + hexadecimal + "+" + binary_exponent + float_suffixes + "?|" +
        hexadecimal_prefix + hexadecimal + r"*\." + hexadecimal + "+" + binary_exponent + float_suffixes + "?|" +
        hexadecimal_prefix + hexadecimal + r"+\." + binary_exponent + float_suffixes + "?" +
        ")"
    )

    t_STRING_LITERAL = (
        "(" + string_prefixes + r'?\"([^"\\\n]|' + escape_sequence + r')*\"' + whitespace + "*)+"
    )

    t_OPERATOR = (
        "(" +
        r"\.\.\.|" +
        separator + "|" +
        symbol + "+" +
        ")"
    )

    t_WHITESPACE = whitespace

    def build(self, **kwargs):
        self.lexer = lex.lex(object=self, **kwargs)

    def input(self, text):
        self.lexer.input(text)

    def token(self):
        self.last_token = self.lexer.token()
        return self.last_token

    def t_error(self, t):
        msg = 'Illegal character %s' % repr(t.value[0])



if __name__ == "__main__":
    lexer = CLexer()
    lexer.build()

    with open(sys.argv[1], 'r') as f:
        text = f.read()

    lexer.input(text)
    while True:
        tok = lexer.token()
        if tok is None:
            break

        if tok.type != 'WHITESPACE':
            print(tok)
