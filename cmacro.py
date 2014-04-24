#!/usr/bin/env python

from __future__ import print_function

import re
import sys
import argparse
from pprint import pprint
from bisect import bisect_left
from collections import namedtuple


###############################################################################
## LEXING

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


###############################################################################
## ABSTRACT SYNTAX TREE


class ASTNode(object):
    """ Root object for all AST nodes.
    """

    def __init__(self, token, parent=None, parent_index=-1):
        # Token represented by this AST.  Can be None.
        self.token        = token

        # Parent pointer
        self.parent       = parent

        # Index of this node in the parent's children array.  This is a helpful
        # but non-required aid for when a macro requires us to touch a given
        # node's siblings - which, as it turns out, is pretty common.
        self.parent_index = parent_index

    def clone(self):
        """ Clone this AST, returning an entirely new node that is a copy of
            this one.
        """
        new_self = ASTNode(self.token, self.parent, self.parent_index)
        return new_self

    def replace_with(self, other):
        """ Replace this node with another node.  Doesn't change the
            identity of this node, but makes it an exact copy of
            another node.
        """
        self.token        = other.token
        self.parent       = other.parent
        self.parent_index = other.parent_index

    def to_tokens(self):
        """ Returns an array of tokens that this AST node represents.
        """
        return [self.token]

    @property
    def type(self):
        if self.token is None:
            return None
        return self.token.type

    @property
    def value(self):
        if self.token is None:
            return None
        return self.token.value

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.token)


INVERSE_BLOCK_TOKEN = {
    '(': ')',
    ')': '(',
    '[': ']',
    ']': '[',
    '{': '}',
    '}': '{',
}


class BlockNode(ASTNode):
    """ A type of AST node that can contain children.
    """
    def __init__(self, token, *args, **kwargs):
        ASTNode.__init__(self, token, *args, **kwargs)

        # Children contained by this node.
        self.children = []

    def to_tokens(self):
        ret = []

        if self.token is not None:
            ret.append(self.token)

        for child in self.children:
            ret.extend(child.to_tokens())

        # This is the implicit ending token for the block.
        if self.token is not None:
            ret.append(Token('OPERATOR',
                             INVERSE_BLOCK_TOKEN[self.token.value],
                             -1,
                             -1))

        return ret


def make_ast(tokens):
    """ Make an AST from a sequence of tokens.
    """
    OPENING_SEPARATORS = set("([{")
    CLOSING_SEPARATORS = set(")]}")

    root = BlockNode(None)

    curr = root
    for token in tokens:
        if token.type == 'OPERATOR' and token.value in OPENING_SEPARATORS:
            new_node = BlockNode(token, curr, len(curr.children))
            curr.children.append(new_node)
            curr = new_node
        elif token.type == 'OPERATOR' and token.value in CLOSING_SEPARATORS:
            # Don't add a node - the block node will insert an implicit ending token.
            curr = curr.parent
        else:
            new_node = ASTNode(token, curr, len(curr.children))
            curr.children.append(new_node)

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

        if isinstance(x, BlockNode):
            print("BLOCK: %s" % (x.token.value,))
            print_ast(x, depth + 1, seen)
        else:
            print("%s: %s" % (x.token.type, x.token.value))


###############################################################################
## MACRO PROCESSING


# TODO: add line number and column number here
class MacroError(Exception):
    pass


class MacroNode(ASTNode):
    """ Special sentinel node for macros.  Will throw an error if
        it's converted to tokens, since this should not be placed
        into the output AST.
    """
    def __init__(self, specifiers):
        self.token = None
        self.parent = None
        self.parent_index = None

        # The specifier is a single word, followed by some number of other
        # words that act as filters.
        self.name = specifiers[0]
        self.filters = specifiers[1:]


class MacroInstance(object):
    """ This class represents a single 'case' arm of a macro, and the
        associated information - match template, replacement template, and
        (optionally) any toplevel tokens to add.
    """

    def __init__(self, name, match, template, toplevel=None):
        self.name = name

        self.matches = []

        # Process match array.  We need to replace a sequence of nodes that are
        # of the form '$ ( IDENTIFIER )' with a special 'macro' node.
        for i in range(len(match)):
            op_node = match[i]

            if (op_node.type == 'OPERATOR' and op_node.value == '$'):
                block_node = match[i + 1]

                if (isinstance(block_node, BlockNode) and
                    block_node.value == '(' and
                    len(block_node.children) > 0 and
                    all(x.type == 'IDENTIFIER' for x in block_node.children)):

                    # Matches!
                    specifiers = [x.value for x in block_node.children]
                    new_node = MacroNode(specifiers)


def process_macros(ast):
    # In order to find macros, we need to look for an identifier 'macro',
    # followed by the identifier 'unless', followed by a {} block.  We walk the
    # entire AST and, for each of them, process the macros for that block.
    # Note that:
    #   - Macros are block scoped, like variables - they can only be used in
    #     the block they're defined in or any lower block.
    #   - Macros are processes from inside out, until they no longer have any
    #     macros.  This means that in the following:
    #
    #       macro foo {
    #           case {
    #               match { $(test) }
    #               template { bar( $(test) ) }
    #           }
    #       }
    #       macro bar {
    #           case {
    #               match { $(test) }
    #               template { if( $(test) ) }
    #           }
    #       }
    #       macro baz {
    #           case {
    #               match { $(test) }
    #               template { $(test) }
    #           }
    #       }
    #
    #
    #       baz {
    #           foo(true) {
    #               a = 1;
    #           }
    #       }
    #
    # The expansion will be as follows:
    #
    #       baz {
    #           bar( (true) ) {
    #               a = 1;
    #           }
    #       }
    #
    # And then:
    #
    #       baz {
    #           if( ( (true) ) ) {
    #               a = 1;
    #           }
    #       }
    #
    # And then:
    #
    #       {
    #           if( ( (true) ) ) {
    #               a = 1;
    #           }
    #       }
    #
    # As you can see, inner macros are recursively expanded until there are no
    # more expansions to be performed, and then macros from higher levels are
    # applied (and so on, up to the root of the file).

    # This is a helper function to parse a single 'case' arm of a macro.
    def parse_case(block):
        # The valid components are:
        #   match {}
        #   template {}
        #   toplevel {}
        match_block = None
        template_block = None
        toplevel_block = None

        for i in range(0, len(block.children), 2):
            id_node = block.children[i]
            block_node = block.children[i + 1]

            if (id_node.type != 'IDENTIFIER' or
                id_node.value not in ['match', 'template', 'toplevel']):
                raise MacroError(
                    "Unknown %s token in case arm: %s" % (id_node.type,
                                                          id_node.value)
                )

            if not (isinstance(block_node, BlockNode) and
                    block_node.value == '{'):
                raise MacroError(
                    "Token following '%s' definition must be a curly brace "
                    "block, not a %s node: %s" % (id_node.value,
                                                  block_node.type,
                                                  block_node.value)
                )

            if 'match' == id_node.value:
                match_block = block_node
            elif 'template' == id_node.value:
                template_block = block_node
            elif 'template' == id_node.value:
                toplevel_block = block_node

        # We must have a match and template blocks.
        if match_block is None:
            raise MacroError("No 'match' block found in case arm")
        if template_block is None:
            raise MacroError("No 'template' block found in case arm")

        return match_block, template_block, toplevel_block

    # This is a helper function to parse a macro block.  It will return a set
    # of tuples (name, (match, template, toplevel))
    def parse_macro(name, block):
        macro_def = []

        # Each item in the block should be a 'case' identifier, followed by a
        # curly brace block.
        for i in range(0, len(block.children), 2):
            case_node = block.children[i]
            block_node = block.children[i + 1]
            if case_node.type != 'IDENTIFIER' or case_node.value != 'case':
                raise MacroError(
                    "Unknown %s token in macro definition: %s" % (
                        case_node.type, case_node.value)
                )

            if not (isinstance(block_node, BlockNode) and
                    block_node.value == '{'):
                raise MacroError(
                    "Token following 'case' definition must be a curly brace "
                    "block, not a %s node: %s" % (block_node.type,
                                                  block_node.value)
                )

            # All good.  Parse this 'case' node.
            case_def = parse_case(block_node)
            macro_def.append((name, case_def))

        return macro_def

    # This is a helper function that, given a sequence of macro tuples in the
    # form (match, template, toplevel), will return all those that match the
    # given tokens.
    def find_match(macros, block, offset):
        for match, _, _ in macros:
            for i in range(offset, len(block)):
                pass

        # TODO
        return False

    # This function will walk the AST and actually process macros.  The
    # 'macros' parameter is a list of lists, where each top-level list is all
    # the macros in a given block, and the contents are macro objects.
    def walk_ast(ast, macros):
        # The macros for this level.
        curr_macros = []

        # Helper function to look for macros in this scope, and then all higher
        # scopes.
        def find_macro(search):
            found = []

            for name, macro in curr_macros:
                if name == search:
                    found.append(macro)

            # Need to start from the end, since inner macros can shadow outer
            # ones.
            for level in reversed(macros):
                for name, macro in level:
                    if name == search:
                        found.append(macro)

            return found


        i = 0
        while True:
            if i == len(ast.children):
                break

            node = ast.children[i]

            # If this is a 'macro' identifier, we need to parse this macro.
            if 'IDENTIFIER' == node.type and 'macro' == node.value:
                # The next token should be an identifier that's the name.
                name_node = ast.children[i + 1]
                if name_node.type != 'IDENTIFIER':
                    raise MacroError(
                        "Expected token after 'macro' to be an identifier, but"
                        " found a '%s' token instead: %s" % (name_node.type,
                                                             name_node.value)
                    )

                # The token after both should be a curly-braced block.
                block_node = ast.children[i + 2]
                if not (isinstance(block_node, BlockNode) and
                        block_node.value == '{'):
                    raise MacroError(
                        "Expected token after macro name to be a block, but "
                        "found a '%s' token instead: %s" % (block_node.type,
                                                            block_node.value)
                    )

                # If we get here, the macro is formed correctly.  We need to
                # remove it from the AST so it doesn't get pretty-printed.
                ast.children.pop(i + 2)
                ast.children.pop(i + 1)
                ast.children.pop(i)

                # Process this macro node.
                macro_defs = parse_macro(name_node.value, block_node)
                curr_macros.extend(macro_defs)

                # Back i up by 1, so that we re-process this position in the
                # next loop (since the node was removed).
                i -= 1

            # Otherwise, check if this is a call to a macro.
            elif 'IDENTIFIER' == node.type:
                possible_matches = find_macro(node.value)
                if len(possible_matches):
                    print("Found call to macro: %s" % (node.value,))

                    # This is an invokation of a macro that has been defined,
                    # and 'macros' now contains all possible matches for that
                    # macro.  We need to find the appropriate match.
                    macro = find_match(possible_matches, ast.children, i + 1)
                    if macro is None:
                        raise MacroError(
                            "Did not find a matching case for the macro '%s' "
                            "(out of %d possible cases)" % (
                                node.value, len(possible_matches))
                        )



            # Otherwise, if this is a block, we need to recurse into it.
            elif isinstance(node, BlockNode) and node.value == '{':
                walk_ast(node, macros + [curr_macros])

            # Otherwise, do nothing.
            i += 1

    walk_ast(ast, [])

    return


###############################################################################
## PRETTY PRINTING


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


###############################################################################
## USER INTERFACE


def lex_c(args):
    lexer = build_c_lexer()
    lexer.input(text)

    tokens = list(lexer.tokens())
    for t in tokens:
        print("%s\t%d\t%d\t%s" % (
            t.type,
            t.line,
            t.col,
            t.value,
        ))

    ast = make_ast(tokens)
    print("-" * 50)
    print("Before macro expansion:")
    print("-" * 50)
    print_ast(ast)

    process_macros(ast)

    print("-" * 50)
    print("After macro expansion:")
    print("-" * 50)
    print_ast(ast)

    print_to_c(ast.to_tokens())


def do_expand(args):
    fname, contents = get_file(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)

    process_macros(ast)

    print_to_c(ast.to_tokens())


def do_lex(args):
    fname, contents = get_file(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    for t in lexer.tokens():
        print("%s\t%d\t%d\t%s" % (
            t.type,
            t.line,
            t.col,
            t.value,
        ))


def get_file(args):
    fname = args.file

    if '-' == fname:
        fname = "<stdin>"
        text = sys.stdin.read()
    else:
        with open(fname, 'r') as f:
            text = f.read()

    return fname, text


def main():
    parser = argparse.ArgumentParser(description='Macro processor for C')
    subparsers = parser.add_subparsers()

    # Global options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('file', type=str, help='file to process')

    # Add parser for 'lex' command, which just prints the tokens to stdout.
    lex_parser = subparsers.add_parser('lex', parents=[parent_parser],
                                       help='lex the input file, and print '
                                            'tokens to stdout')
    lex_parser.set_defaults(func=do_lex)


    # Add parser for 'expand' command, which actually expands macros
    expand_parser = subparsers.add_parser('expand', parents=[parent_parser],
                                          help='expand macros in the input')
    expand_parser.set_defaults(func=do_expand)

    # Parse arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
