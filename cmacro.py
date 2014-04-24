#!/usr/bin/env python

from __future__ import print_function

import re
import sys
from bisect import bisect_left
from collections import namedtuple


# This is our token class - it represents a single token in the input
# file, along with its value, line and column.
Token = namedtuple('Token', 'type value line col')


class LexerError(Exception):
    def __init__(self, line, col):
        self.line = line
        self.col  = col


class Lexer(object):
    """ A simple regex-based lexer/tokenizer.
    """
    def __init__(self, rules, skip_whitespace=True):
        """ Create a lexer.

            rules:
                A list of rules. Each rule is a regex, type
                pair, where regex is the regular expression used
                to recognize the token and type is the type
                of the token to return when it's recognized.

            skip_whitespace:
                If True, whitespace (\s+) will be skipped and not
                reported by the lexer. Otherwise, you have to
                specify your rules for whitespace, or it will be
                flagged as an error.
        """
        # All the regexes are concatenated into a single one
        # with named groups. Since the group names must be valid
        # Python identifiers, but the token types used by the
        # user are arbitrary strings, we auto-generate the group
        # names and map them to token types.
        regex_parts = []
        self.group_type = {}

        for idx, (regex, type) in enumerate(rules):
            groupname = 'GROUP%s' % idx
            regex_parts.append('(?P<%s>%s)' % (groupname, regex))
            self.group_type[groupname] = type

        self.regex = re.compile('|'.join(regex_parts))
        self.skip_whitespace = skip_whitespace
        self.re_ws_skip = re.compile('\S')

    def input(self, buf, line_ending='\n'):
        """ Initialize the lexer with a buffer as input.
        """
        self.buf = buf
        self.pos = 0

        # Search through the input and build a mapping of positions --> lines,
        # with a virtual newline in the -1th position (i.e. at the start of the
        # file).
        matches = re.finditer(re.escape(line_ending), buf)
        self.newlines = [-1] + sorted(m.start() for m in matches)

    @property
    def line(self):
        """ Returns the current line number that we are at, as
            determined by the value of self.pos
        """

        # We use the bisection algorithm to find the location of this position
        # in our array of newlines.  bisect_left will return the position that
        # is lower than any position in the array, and thus the line number
        # (starting from 0).  However, since we've added the virtual newline
        # above, the 0th entry in the array corresponds to the first line, and
        # thus we don't need to do anything extra.
        return bisect_left(self.newlines, self.pos)

    @property
    def col(self):
        """ Returns the current column number that we are at, as
            determined by the value of self.pos
        """

        # We need to find the line number...
        line = self.line

        # ... and use this to find the start of this line.  We subtract one
        # from the current line number to get the position of the end of the
        # previous line, which we then subtract from our position to get the
        # column number.  Note that the subtraction includes the newline, which
        # is okay since we want a 1-indexed column number.  For a 0-indexed
        # value, we should subtract 1 to handle the newline.
        return self.pos - self.newlines[line - 1]

    def token(self):
        """ Return the next token (a Token object) found in the
            input buffer. None is returned if the end of the
            buffer was reached.
            In case of a lexing error (the current chunk of the
            buffer matches no rule), a LexerError is raised with
            the position of the error.
        """
        if self.pos >= len(self.buf):
            return None
        else:
            if self.skip_whitespace:
                m = self.re_ws_skip.search(self.buf, self.pos)

                if m:
                    self.pos = m.start()
                else:
                    return None

            m = self.regex.match(self.buf, self.pos)
            if m:
                groupname = m.lastgroup
                tok_type = self.group_type[groupname]
                tok = Token(tok_type, m.group(groupname), self.line, self.col)
                self.pos = m.end()
                return tok

            # if we're here, no rule matched
            raise LexerError(self.line, self.col)

    def tokens(self):
        """ Returns an iterator to the tokens found in the buffer.
        """
        while 1:
            tok = self.token()
            if tok is None: break
            yield tok


def build_c_lexer():
    # Helper components
    INT_SUFFIXES = "(((u|U)(l|L|ll|LL)?)|((l|L|ll|LL)(u|U)?))"
    ESCAPE_SEQUENCE = r"""
        (\\(['"\?\\abfnrtv]|[0-7]{1,3}|x[a-fA-F0-9]+))
    """.strip()
    EXPONENT = '([Ee][+-]?[0-9]+)'
    BEXPONENT = '([Pp][+-]?[0-9]+)'

    # The actual rules for lexing
    rules = [
        # Identifiers
        (r'[a-zA-Z_][a-zA-Z0-9_]*',                             'IDENTIFIER'),

        # Floating point numbers
        #   1)  1234e1
        #   2)  .234
        #   3)  0x1ap3b
        #   4)  0x.13abp3b
        #   5)  0x1a.p3b
        (r'[0-9]+' + EXPONENT + '(f|F|l|L)?',                   'F_CONSTANT'),
        (r'[0-9]*\.[0-9]+' + EXPONENT + '?(f|F|l|L)?',          'F_CONSTANT'),
        (r'0[xX][a-fA-F0-9]+' + BEXPONENT + '(f|F|l|L)?',       'F_CONSTANT'),
        (r'0[xX][a-fA-F0-9]*\.[a-fA-F0-9]+' + BEXPONENT +
            r'(f|F|l|L)?',                                      'F_CONSTANT'),
        (r'0[xX][a-fA-F0-9]+\.' + BEXPONENT + '(f|F|l|L)?',     'F_CONSTANT'),

        # Integers
        #   1)  0x123abc
        #   2)  1234
        #   3)  0777
        #   4)  'q'
        (r'0[xX][a-fA-F0-9]+' + INT_SUFFIXES + '?',             'I_CONSTANT'),
        (r'[1-9][0-9]*' + INT_SUFFIXES + '?',                   'I_CONSTANT'),
        (r'0[0-7]*' + INT_SUFFIXES + '?',                       'I_CONSTANT'),
        (r"(u|U|L)?\'([^'\\\n]|" + ESCAPE_SEQUENCE + r")+\'",   'I_CONSTANT'),

        # String literals
        (r'L?\"(\\.|[^\\"])*\"',                                'STRING'),

        # Operators
        #   1)  ...
        #   2) , ; () [] {} .
        #   3) Anything not in above sets
        (r'\.\.\.',                                             'OPERATOR'),
        (r'[\,\;\(\)\[\]\{\}\.]',                               'OPERATOR'),
        (r'[^ \t\v\n\f\,\;\(\)\[\]\{\}\.a-zA-Z_0-9]+',          'OPERATOR'),
    ]

    lexer = Lexer(rules)
    return lexer


class ASTNode(object):
    """ Root object for all AST nodes.

        TODO: split this apart into individual types of nodes?
    """

    def __init__(self, token, parent=None, parent_index=-1):
        self.token        = token
        self.children     = []

        # Parent pointer
        self.parent       = parent

        # Index of this node in the parent's children array.  This is a helpful
        # but non-required aid for when a macro requires us to touch a given
        # node's siblings - which, as it turns out, is pretty common.
        self.parent_index = parent_index

    def clone(self):
        """ Clone this AST and all subnodes, returning an entirely new node
            that is a copy of this one.
        """
        new_self = ASTNode(self.token, self.parent)
        new_self.children.extend(child.clone() for child in self.children)
        return new_self

    def _internal_flatten(self, ret):
        for node in self.children:
            ret.append(node.token)

            if len(node.children):
                node._internal_flatten(ret)

        return ret

    def flatten(self, include_self=False):
        """ Flatten this AST and all subnodes into a token array.
        """
        ret = []
        if include_self:
            ret.append(self.token)

        return self._internal_flatten(ret)

    def replace_with(self, other):
        """ Replace this node with another node.  Doesn't change the
            identity of this node, but makes it an exact copy of
            another node.
        """
        self.token        = other.token
        self.children     = other.children
        self.parent       = other.parent
        self.parent_index = other.parent_index

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.token)


# TODO: This isn't really an AST - we have all the individual tokens here,
#       just organized as a tree structure.  There's probably a better way.
def make_ast(tokens):
    """ Make an AST from a sequence of tokens.
    """
    OPENING_SEPARATORS = set("([{")
    CLOSING_SEPARATORS = set(")]}")

    root = ASTNode(None)

    curr = root
    for token in tokens:
        new_node = ASTNode(token, curr, len(curr.children))
        curr.children.append(new_node)

        # Move up or down the scope hierarchy as necessary.
        if token.type == 'OPERATOR' and token.value in OPENING_SEPARATORS:
            curr = new_node
        elif token.type == 'OPERATOR' and token.value in CLOSING_SEPARATORS:
            curr = curr.parent

    return root


def print_ast(ast, depth=0, seen=None):
    """ Print an AST as a formatted tree.
    """
    if seen is None:
        seen = set()

    if ast in seen:
        print("** loop: seen: %s" % (ast,))
        return

    seen.add(ast)

    for x in ast.children:
        if depth > 0:
            print("-" * depth + ">", end='')
        print("%s: %s" % (x.token.type, x.token.value))

        if len(x.children) > 0:
            print_ast(x, depth + 1, seen)


def process_macros(ast):
    """ Given an AST, process macros within it.
    """

    def find_in_ast(node, cond, seen):
        # If we've already seen this node, stop
        # TODO: should error - no loops allowed
        if node in seen:
            return
        seen.add(node)

        # Is this node what we're looking for?
        if node.token is not None and cond(node):
            yield node

        # For each child of this node, recurse to it.
        for child in node.children:
            yield from find_in_ast(child, cond, seen)

    for unode in find_in_ast(ast, lambda n: n.token.type == 'IDENTIFIER' and
                                            n.token.value == 'unless',
                             set()):

        # Now, the first sibling of the 'unless' node should be an operator (
        # node, representing the condition.
        cond_node = unode.parent.children[unode.parent_index + 1]
        if cond_node.token.type != 'OPERATOR' or cond_node.token.value != '(':
            print('ERROR EXPANDING: %s is not a "(" node' % (cond_node,))
            break

        # Replace the 'unless' with an 'if'.
        if_node = ASTNode(Token('IDENTIFIER', 'if', -1, -1),
                          unode.parent,
                          unode.parent_index)
        unode.replace_with(if_node)

        # Get the condition node, and replace:
        #   operator(
        #       children
        #       )
        # With:
        #   operator(
        #       operator !
        #       operator(
        #           children
        #           )
        #       )

        new_cond_node = ASTNode(Token('OPERATOR', '(', -1, -1))
        new_cond_node.children.append(ASTNode(Token('OPERATOR', '!', -1, -1)))
        new_cond_node.children.append(cond_node)
        new_cond_node.children.append(ASTNode(Token('OPERATOR', ')', -1, -1)))

        unode.parent.children[unode.parent_index + 1] = new_cond_node


def print_to_c(tokens):
    # Simple pretty-printing rules:
    #   - If it's an opening curly brace, print a newline and increment depth
    #   - If is's a closing curly brace, print a newline and decrement depth
    #   - If it's a semicolon, print a newline unless we're in brackets.
    #   - Don't print a space if:
    #       - The next item is a comma or closing brace.
    #       - The curren item is an opening brace.
    #
    # TODO: Rework printing cycle so we don't have closing curly braces
    #       indented to the level of the inner block.
    # TODO: Have a feature for adding #line directives, or similar, so we can
    #       link back to the original source file.

    block_depth = 0
    brace_depth = 0

    for i, tok in enumerate(tokens):
        print(tok.value, end='')
        if ((i < len(tokens) - 1 and tokens[i + 1].value not in [',', ')']) and
            (tok.value != '(')):
            print(' ', end='')

        if tok.value == '(':
            brace_depth += 1
        elif tok.value == ')':
            brace_depth -= 1

        if tok.value in ['{', '}', ';']:
            if tok.value == '{':
                block_depth += 1
            elif tok.value == '}':
                block_depth -= 1

            if brace_depth == 0:
                print('\n' + ('    ' * block_depth), end='')


def lex_c(text):
    lexer = build_c_lexer()
    lexer.input(text)

    tokens = list(lexer.tokens())
    # for t in tokens:
    #     print(t)

    ast = make_ast(tokens)
    print("Before macro expansion:")
    print("-" * 50)
    print_ast(ast)

    process_macros(ast)

    print("After macro expansion:")
    print("-" * 50)
    print_ast(ast)

    print_to_c(ast.flatten())


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        text = f.read()
    lex_c(text)
