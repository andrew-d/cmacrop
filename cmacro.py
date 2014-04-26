#!/usr/bin/env python

from __future__ import print_function

import re
import abc
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


class ASTNode(metaclass=abc.ABCMeta):
    """ Root object for all AST nodes.
    """

    def __init__(self, token, parent=None):
        self.__token = token
        self.__parent = parent

    @property
    def token(self):
        return self.__token

    @property
    def parent(self):
        return self.__parent

    @property
    def next(self):
        """ Returns the next (sibling) node in the AST.
        """
        if self.parent is None:
            return None

        children = self.parent.children
        try:
            idx = children.index(self)
        except ValueError:
            return None

        if idx == len(children) - 1:
            return None

        return children[idx + 1]

    @property
    def previous(self):
        """ Returns the previous (sibling) node in the AST.
        """
        if self.parent is None:
            return None

        children = self.parent.children
        try:
            idx = children.index(self)
        except ValueError:
            return None

        if idx == 0:
            return None

        return children[idx - 1]

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.token)


class IdentifierNode(ASTNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__identifier = token.value

    @property
    def identifier(self):
        return self.__identifier


class ConstantNode(ASTNode):
    pass


class IntConstantNode(ConstantNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__value = token.value

    @property
    def value(self):
        return self.__value


class FloatConstantNode(ConstantNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__value = token.value

    @property
    def value(self):
        return self.__value


class StringNode(ASTNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__value = token.value

    @property
    def value(self):
        return self.__value


class OperatorNode(ASTNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__operator = token.value

    @property
    def operator(self):
        return self.__operator


class BlockNode(ASTNode):
    """ A type of AST node that can contain children.
    """
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        # Children contained by this node.
        self.__children = []

    @property
    def children(self):
        return self.__children

    @property
    def type(self):
        return self.token.value


class RootNode(BlockNode):
    """ A special block node that represents the root of the AST.  Hides the
        'type' field from the parent BlockNode.
    """
    def __init__(self):
        BlockNode.__init__(self, token=None, parent=None)

    @property
    def type(self):
        return None


class MacroNode(ASTNode):
    """ Special sentinel node for macros.
    """
    def __init__(self, name, filters):
        ASTNode.__init__(self, token=None, parent=None)

        # The specifier is a single word, followed by some number of other
        # words that act as filters.
        self.__name = name
        self.__filters = filters

    @property
    def name(self):
        return self.__name

    @property
    def filters(self):
        return self.__filters


class NodeVisitor(object):
    """ A node visitor that walks each node in the AST and calls visitor
        methods for every node found.

        Modeled after the Python standard library's ast.NodeVisitor
    """
    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if not isinstance(node, BlockNode):
            return

        for child in node.children:
            self.visit(child)


class NodeTransformer(NodeVisitor):
    """ A subclass of the node visitor that allows modifying the AST as it's
        traversed.  Returning a value from a visitor function will replace that
        node in the AST with the returned value.

        Modeled after the Python standard library's ast.NodeTransformer
    """

    def generic_visit(self, node):
        if not isinstance(node, BlockNode):
            return node

        new_children = []
        for child in node.children:
            value = self.visit(child)
            if value is None:
                continue
            elif not isinstance(value, ASTNode):
                new_children.extend(value)
                continue

            new_children.append(value)

        node.children[:] = new_children
        return node


def make_ast(tokens):
    """ Make an AST from a sequence of tokens.
    """
    OPENING_SEPARATORS = set("([{")
    CLOSING_SEPARATORS = set(")]}")

    root = RootNode()

    curr = root
    for token in tokens:
        if token.type == 'OPERATOR':
            if token.value in OPENING_SEPARATORS:
                new_node = BlockNode(token, curr)
                curr.children.append(new_node)
                curr = new_node

            elif token.value in CLOSING_SEPARATORS:
                # Don't add a node - the parent block node will insert an
                # implicit ending token when we're re-serializing the AST.
                curr = curr.parent

            else:
                # Create an operator node.
                curr.children.append(OperatorNode(token, curr))

        elif token.type == 'IDENTIFIER':
            curr.children.append(IdentifierNode(token, curr))

        elif token.type == 'STRING':
            curr.children.append(StringNode(token, curr))

        elif token.type == 'I_CONSTANT':
            curr.children.append(IntConstantNode(token, curr))

        elif token.type == 'F_CONSTANT':
            curr.children.append(FloatConstantNode(token, curr))

        else:
            raise RuntimeError("Unknown token type: %s" % (token.type,))

    return root


class ASTPrinter(NodeVisitor):
    def __init__(self):
        self.seen = set()
        self.depth = 0

    def generic_visit(self, node):
        if node.token is not None:
            if self.depth > 0:
                print(('-' * self.depth) + '>', end='')

            print('%s: %s' % (node.token.type, node.token.value))

        self.depth += 1
        NodeVisitor.generic_visit(self, node)
        self.depth -= 1


def print_ast(ast):
    """ Print an AST as a formatted tree.
    """
    ASTPrinter().visit(ast)


###############################################################################
## MACRO PROCESSING


# TODO: add line number and column number here
class MacroError(Exception):
    pass


class MacroStripper(NodeTransformer):
    def __init__(self):
        # Format [(name, block), ...]
        self.macro_blocks = []
        self.strip = 0

    def check_strip(self):
        if self.strip > 0:
            self.strip -= 1
            return True

        return False

    def visit_IdentifierNode(self, node):
        if self.check_strip():
            return None

        if node.identifier != 'macro':
            return node

        name_node = node.next
        if name_node is None:
            raise MacroError("No macro name found after 'macro' token")

        if not isinstance(name_node, IdentifierNode):
            err = ("Expected token after 'macro' to be an identifier, but "
                   "found a '%s' token instead: %s")
            raise MacroError(err % (name_node.token.type,
                                    name_node.token.value))

        bnode = name_node.next
        if bnode is None:
            raise MacroError("No block found after macro name")

        if not (isinstance(bnode, BlockNode) and bnode.type == '{'):
            err = ("Expected token after name to be a block, but found a '%s' "
                   "token instead: %s")
            raise MacroError(err % (bnode.token.type, bnode.token.value))

        # Add this macro to the return array.
        self.macro_blocks.append((name_node.identifier, bnode))

        # Set the 'strip from AST' flag which indicates that we should remove
        # the next two nodes from the AST.
        self.strip = 2

    def visit_BlockNode(self, node):
        if self.check_strip():
            return None

        # If we're not removing this block, we recurse into it.
        return self.generic_visit(node)

    def visit_RootNode(self, node):
        # Root nodes are also blocks.
        return self.generic_visit(node)


def strip_macros(ast):
    """ Strip all macro definitions from the AST and return them as a seperate
        array, along with the AST itself (now lacking macros).
    """
    sp = MacroStripper()
    sp.visit(ast)

    return ast, sp.macro_blocks


def is_macro_visible_from(macro_node, block):
    """ Returns whether a macro node is visible from a given block, by walking
        the parent chain of the block and checking if the block the macro is
        defined in is in it.
        TODO: return distance too?
    """
    def_block = macro_node.parent
    par = block
    while par is not None:
        if par is def_block:
            return True

        par = par.parent

    return False


class CaseVisitor(NodeVisitor):
    def __init__(self):
        self.block_ty = None
        self.blocks = {}

    def visit_IdentifierNode(self, node):
        # If we've hit the identifier already, it means we have something like
        # 'match' 'match', with no intervening block.
        if self.block_ty is not None:
            raise MacroError("Unexpected identifier '%s'" % (node.identifier,))

        # Check for valid identifier
        if node.identifier not in ['match', 'template', 'toplevel']:
            raise MacroError(
                "Unknown identifier token in case arm: %s" % (
                   node.identifier,))

        # Save this type.
        self.block_ty = node.identifier

    def visit_BlockNode(self, node):
        # If we get here, the block type should be set (indicating that we've
        # seen a proper identifier before).
        if self.block_ty is None:
            raise MacroError("Unexpected block token")

        # Great.  Save all this.
        self.blocks[self.block_ty] = node

        # Reset the block type so that we don't error out.
        self.block_ty = None

    def generic_visit(self, node):
        # Something that's not a block or identifier node is an error.
        if node.token is None:
            ty = 'UNKNOWN'
            val = 'UNKNOWN'
        else:
            ty = node.token.type
            val = node.token.value

        raise MacroError("Unknown %s token in case arm: %s" % (ty, val))


def parse_case(block):
    """ Parse a 'case {}' block inside a macro definition into the match,
        template and (optionally) toplevel blocks.
    """

    visitor = CaseVisitor()
    visitor.visit(block)

    blocks = visitor.blocks

    # We must have a match and template blocks.
    if 'match' not in blocks:
        raise MacroError("No 'match' block found in case arm")
    if 'template' not in blocks:
        raise MacroError("No 'template' block found in case arm")

    return blocks


class MacroVisitor(NodeVisitor):
    def __init__(self):
        self.cases = []
        self.root = True
        self.found_case = False

    def visit_IdentifierNode(self, node):
        if self.found_case:
            raise MacroError("'case' found without intervening block")

        if node.identifier != 'case':
            raise MacroError("Unknown identifier token in macro definition: "
                             "%s" % (node.identifier,))

        self.found_case = True

    def visit_BlockNode(self, node):
        if self.root:
            self.root = False
            return NodeVisitor.generic_visit(self, node)

        if not self.found_case:
            raise MacroError("block found without preceding 'case'")
        if node.type != '{':
            raise MacroError("Case blocks must be delimited with curly braces")

        self.cases.append(node)
        self.found_case = False


def parse_single_macro(block):
    visitor = MacroVisitor()
    visitor.visit(block)

    # TODO: need to modify the cases and turn '$' '(' <block> ')' into a
    #       new macro node

    return visitor.cases


def parse_macros(macros):
    ret = []
    for name, macro_block in macros:
        ret.append((name, parse_single_macro(macro_block)))

    return ret


def process_macros(ast):
    ast, macros = strip_macros(ast)
    parsed = parse_macros(macros)

    for name, cases in parsed:
        print(name)
        for case in cases:
            print('-' * 50)
            print_ast(case)
        print('-' * 50)

    return ast


def process_macros_old(ast):
    # In order to find macros, we need to look for an identifier 'macro',
    # followed by an identifier, e.g. 'unless', followed by a {} block.  We
    # walk the entire AST and, for each of them, process the macros for that
    # block.
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
    # Also note that macro DEFINITIONS aren't processed - we expand macros only
    # at the site of usage - the definition is inserted as-is in to the AST,
    # and then expanded until no child macros are found, and so on.

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


INVERSE_BLOCK_TOKEN = {
    '(': ')',
    ')': '(',
    '[': ']',
    ']': '[',
    '{': '}',
    '}': '{',
}


class TokenGenerator(NodeVisitor):
    def __init__(self):
        self.tokens = []

    def generic_visit(self, node):
        if node.token is not None:
            self.tokens.append(node.token)

        NodeVisitor.generic_visit(self, node)

        if node.token is not None and isinstance(node, BlockNode):
            inverse = INVERSE_BLOCK_TOKEN[node.token.value]
            self.tokens.append(Token('OPERATOR', inverse, -1, -1))


def to_tokens(ast):
    gen = TokenGenerator()
    gen.visit(ast)
    return gen.tokens


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


def do_expand(args):
    fname, contents = get_file(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)

    process_macros(ast)

    tokens = to_tokens(ast)
    print_to_c(tokens)


def do_print_ast(args):
    fname, contents = get_file(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)

    if args.strip_macros:
        strip_macros(ast)

    print_ast(ast)


def do_print_macros(args):
    fname, contents = get_file(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)
    ast, macros = strip_macros(ast)

    # TODO: parse the macros before printing them

    for name, macro in macros:
        print("Macro '%s'" % (name,))
        print('-' * 50)

        print_ast(macro)


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

    lex_parser = subparsers.add_parser('lex', parents=[parent_parser],
                                       help='lex the input file, and print '
                                            'tokens to stdout')
    lex_parser.set_defaults(func=do_lex)


    expand_parser = subparsers.add_parser('expand', parents=[parent_parser],
                                          help='expand macros in the input')
    expand_parser.set_defaults(func=do_expand)

    print_ast_parser = subparsers.add_parser('print_ast',
                                             parents=[parent_parser],
                                             help='build and print the AST')
    print_ast_parser.add_argument("--strip-macros", action="store_true",
                                  help="strip macros from the AST before "
                                       "printing")
    print_ast_parser.set_defaults(func=do_print_ast)

    print_macros_parser = subparsers.add_parser('print_macros',
                                             parents=[parent_parser],
                                             help='print all found macros')
    print_macros_parser.set_defaults(func=do_print_macros)

    # Parse arguments.
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
