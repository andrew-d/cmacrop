#!/usr/bin/env python

from __future__ import print_function

import re
import abc
import sys
import logging
import argparse
from pprint import pprint
from bisect import bisect_left
from collections import namedtuple


###############################################################################
## UTILITY

# Configure logging
log = logging.getLogger('cmacro')
log.addHandler(logging.NullHandler())


def add_metaclass(metaclass):
    """ Class decorator for creating a class with a metaclass that supports
        both Python 2 and 3.

        Taken with thanks from 'six':
        https://bitbucket.org/gutworth/six/src/4420499da4959da591043652e50bf75f2d3398f7/six.py?at=default
    """
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


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


@add_metaclass(abc.ABCMeta)
class ASTNode(object):
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

    @parent.setter
    def parent(self, parent):
        self.__parent = parent

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
            log.warn("Did not find self in parent.children: %r", self)
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
            log.warn("Did not find self in parent.children: %r", self)
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


class NumConstantNode(ConstantNode):
    pass


class IntConstantNode(NumConstantNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__value = token.value

    @property
    def value(self):
        return self.__value


class FloatConstantNode(NumConstantNode):
    def __init__(self, token, parent=None):
        ASTNode.__init__(self, token, parent)

        self.__value = token.value

    @property
    def value(self):
        return self.__value


class StringNode(ConstantNode):
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


class MacroBindingNode(ASTNode):
    """ Special sentinel node for macro bindings in the form $(name [filters]).
    """
    def __init__(self, name, filters, parent=None):
        ASTNode.__init__(self, token=None, parent=parent)

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


class MacroCommandNode(ASTNode):
    """ Special sentinel node for macro functions in the form
        $(@command [args]).
    """
    def __init__(self, command, args, parent=None):
        ASTNode.__init__(self, token=None, parent=parent)

        self.__command = command
        self.__args = args

    @property
    def command(self):
        return self.__command

    @property
    def args(self):
        return self.__args


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
    def __init__(self, out):
        self.out = out
        self.seen = set()
        self.depth = 0

    def print_depth(self):
        if self.depth > 0:
            print(('-' * self.depth) + '>', end='', file=self.out)

    def visit_MacroBindingNode(self, node):
        self.print_depth()
        print('MACRO BINDING: %s (filters: %r)' % (node.name, node.filters),
              file=self.out)

    def visit_MacroCommandNode(self, node):
        self.print_depth()
        print('MACRO COMMAND: %s (args: %r)' % (node.command, node.args),
              file=self.out)

    def generic_visit(self, node):
        if node.token is not None:
            self.print_depth()
            print('%s: %s' % (node.token.type, node.token.value),
                  file=self.out)

        self.depth += 1
        NodeVisitor.generic_visit(self, node)
        self.depth -= 1


def print_ast(ast, out=sys.stdout):
    """ Print an AST as a formatted tree.
    """
    ASTPrinter(out).visit(ast)


###############################################################################
## MACRO PROCESSING


class MacroError(Exception):
    def __init__(self, msg, line=-1, col=-1):
        self.msg  = 0
        self.line = line
        self.col  = col
        Exception.__init__(self, msg)


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
            raise MacroError("No macro name found after 'macro' token",
                             line=node.token.line, col=node.token.col)

        if not isinstance(name_node, IdentifierNode):
            err = ("Expected token after 'macro' to be an identifier, but "
                   "found a '%s' token instead: %s")
            raise MacroError(err % (name_node.token.type,
                                    name_node.token.value),
                             line=name_node.token.line,
                             col=name_node.token.col)

        bnode = name_node.next
        if bnode is None:
            raise MacroError("No block found after macro name",
                             line=name_node.token.line,
                             col=name_node.token.col)

        if not (isinstance(bnode, BlockNode) and bnode.type == '{'):
            err = ("Expected token after name to be a block, but found a '%s' "
                   "token instead: %s")
            raise MacroError(err % (bnode.token.type, bnode.token.value),
                             line=bnode.token.line, col=bnode.token.col)

        # Add this macro to the return array.
        self.macro_blocks.append((name_node.identifier, bnode))

        # Set the 'strip from AST' flag which indicates that we should remove
        # the next two nodes from the AST (the name and block after the 'macro'
        # keyword).
        self.strip = 2

        # Explicitly remove this node too.
        return None

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
        TODO: return distance too, so we can support overloading macros?
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
        self.root = True
        self.block_ty = None
        self.blocks = {}

    def visit_IdentifierNode(self, node):
        # If we've hit the identifier already, it means we have something like
        # 'match' 'match', with no intervening block.
        if self.block_ty is not None:
            raise MacroError("Unexpected identifier '%s'" % (node.identifier,),
                             line=node.token.line, col=node.token.col)

        # Check for valid identifier
        if node.identifier not in ['match', 'template', 'toplevel']:
            raise MacroError(
                "Unknown identifier token in case arm: %s" % (
                   node.identifier,),
                line=node.token.line,
                col=node.token.col
            )

        # Save this type.
        self.block_ty = node.identifier

    def visit_BlockNode(self, node):
        # The first block seen is the root node, and should be traversed.
        if self.root is True:
            self.root = False
            return NodeVisitor.generic_visit(self, node)

        # If we get here, the block type should be set (indicating that we've
        # seen a proper identifier before).
        if self.block_ty is None:
            raise MacroError("Unexpected block token",
                             line=node.token.line, col=node.token.col)

        # Great.  Save all this.
        self.blocks[self.block_ty] = node

        # Reset the block type so that we don't error out.
        self.block_ty = None

    def generic_visit(self, node):
        # Something that's not a block or identifier node is an error.
        if node.token is None:
            log.warn("Found node with no token: %r", node)
            ty = 'UNKNOWN'
            val = 'UNKNOWN'
        else:
            ty = node.token.type
            val = node.token.value

        raise MacroError("Unknown %s token in case arm: %s" % (ty, val),
                         line=node.token.line, col=node.token.col)


def parse_case(block):
    """ Parse a 'case {}' block inside a macro definition into the match,
        template and (optionally) toplevel blocks.
    """

    visitor = CaseVisitor()
    visitor.visit(block)

    blocks = visitor.blocks

    # We must have a match and template blocks.
    if 'match' not in blocks:
        raise MacroError("No 'match' block found in case arm",
                         line=block.token.line, col=block.token.col)
    if 'template' not in blocks:
        raise MacroError("No 'template' block found in case arm",
                         line=block.token.line, col=block.token.col)

    return blocks


class MacroVisitor(NodeVisitor):
    def __init__(self):
        self.cases = []
        self.root = True
        self.found_case = False

    def visit_IdentifierNode(self, node):
        if self.found_case:
            raise MacroError("'case' found without intervening block",
                             line=node.token.line, col=node.token.col)

        if node.identifier != 'case':
            raise MacroError("Unknown identifier token in macro definition: "
                             "%s" % (node.identifier,),
                             line=node.token.line, col=node.token.col)

        self.found_case = True

    def visit_BlockNode(self, node):
        if self.root:
            self.root = False
            return NodeVisitor.generic_visit(self, node)

        if not self.found_case:
            raise MacroError("block found without preceding 'case'",
                             line=node.token.line, col=node.token.col)
        if node.type != '{':
            raise MacroError("Case blocks must be delimited with curly braces",
                             line=node.token.line, col=node.token.col)

        self.cases.append(node)
        self.found_case = False


VALID_FILTERS = set([
    'ident',
    'int',
    'float',
    'num',
    'string',
    'const',
    'op',
    'list',
    'array',
    'block',
])


class MacroNodeCreator(NodeTransformer):
    def __init__(self):
        self.strip = 0
        self.filters_allowed = False
        self.commands_allowed = False

    def visit_OperatorNode(self, node):
        if node.operator != '$':
            return node

        # The next token should be a bracket block token.
        blk = node.next
        if blk is None:
            log.debug("No node found after '$'")
            return node
        if not (isinstance(blk, BlockNode) and blk.type == '('):
            log.debug("Node after '$' isn't a () block")
            return node

        # Should have at least one child of the () block.
        if len(blk.children) < 1:
            log.debug("No children in () block")
            return node

        # If the first thing in the block is a '@' operator, then this is a
        # command.  Otherwise, it's a binding.
        if (isinstance(blk.children[0], OperatorNode) and
            blk.children[0].operator == '@'):
            log.debug("Found initial macro command at line %d, col %d",
                      node.token.line, node.token.col)

            if not self.commands_allowed:
                raise MacroError("Commands aren't allowed in 'match' blocks")

            if len(blk.children) < 2:
                log.debug("Didn't find a command name after '@'")
                return node
            if not isinstance(blk.children[1], IdentifierNode):
                log.debug("Node after '@' isn't an identifier")
                return node

            command = blk.children[1].identifier
            args    = blk.children[2:]

            log.debug("Creating command node: $(@%s, ...)", command)
            replacement_node = MacroCommandNode(command, args, node.parent)

        else:
            log.debug("Found initial macro binding at line %d, col %d",
                      node.token.line, node.token.col)

            # For bindings, everything should be an identifier.
            if not all(isinstance(x, IdentifierNode) for x in blk.children):
                log.debug("Non-identifier found in () block: %r",
                          [x for x in blk.children
                           if not isinstance(x, IdentifierNode)])
                return node

            names = list(map(lambda n: n.identifier, blk.children))

            # Check for invalid filters.
            if len(names) > 1 and not self.filters_allowed:
                raise MacroError("Filters only allowed when defining "
                                 "variables")

            for flt in names[1:]:
                if flt not in VALID_FILTERS:
                    raise MacroError("Invalid filter found: %s" % (flt,),
                                     line=blk.token.line, col=blk.token.col)

            log.debug("Creating binding node: $(%s, ...)", names[0])
            replacement_node = MacroBindingNode(names[0], names[1:],
                                                node.parent)

        # Success - strip the following block, and return the new node.
        self.strip = 1
        return replacement_node

    def visit_BlockNode(self, node):
        if self.strip > 0:
            self.strip -= 1
            return None

        return self.generic_visit(node)


def parse_single_macro(block):
    visitor = MacroVisitor()
    visitor.visit(block)
    cases = visitor.cases

    # For each case, parse out the set of 'match', 'template' and 'toplevel'
    # blocks.
    all_blocks = []
    for case in visitor.cases:
        all_blocks.append(parse_case(case))

    # For each case, turn '$' '(' <something> ')' into a new macro node.
    tform = MacroNodeCreator()
    for blocks in all_blocks:
        for ty, block in blocks.items():
            is_match = (ty == 'match')
            tform.filters_allowed = is_match
            tform.commands_allowed = not is_match
            tform.visit(block)

    return all_blocks


def parse_macros(macros):
    ret = []
    for name, macro_block in macros:
        # TODO: need to determine how we support multiple definitions of
        #       macros, in case we want to, e.g. allow overloading a macro.
        #       See notes in is_macro_visible_from, esp. regarding distance.
        ret.append((name, parse_single_macro(macro_block)))

    return ret


def clone_ast(ast):
    if isinstance(ast, MacroBindingNode):
        return MacroBindingNode(ast.name, ast.filters[:], ast.parent)

    elif isinstance(ast, MacroCommandNode):
        return MacroCommandNode(ast.command, ast.args[:], ast.parent)

    elif not isinstance(ast, BlockNode):
        return ast.__class__(ast.token, ast.parent)

    else:
        # Is a BlockNode or subclass
        new_node = ast.__class__(ast.token, ast.parent)
        for child in ast.children:
            new_node.children.append(clone_ast(child))
        return new_node


class TemplateReplacer(NodeTransformer):
    def __init__(self, bindings):
        self.bindings = bindings

    def visit_MacroBindingNode(self, node):
        bound = self.bindings[node.name]
        return bound

    def visit_MacroCommandNode(self, node):
        # TODO: run command.
        return IdentifierNode(Token('IDENTIFIER', '<command>', -1, -1))


class MacroApplier(NodeTransformer):
    def __init__(self, macros):
        self.macros = macros
        self.toplevels = []
        self.reset()

        # Sanity checking for FILTER_CLASSES
        for x in self.FILTER_CLASSES.keys():
            if not x in VALID_FILTERS:
                raise RuntimeError("%s not in VALID_FILTERS" % (x,))

    def reset(self):
        self.strip  = 0
        self.changed = False

    def are_nodes_equal(self, node1, node2):
        if node1.__class__ != node2.__class__:
            return False

        if node1.token is not None:
            if node2.token is None:
                return False

            return (node1.token.type == node2.token.type and
                    node1.token.value == node2.token.value)

        if node2.token is not None:
            # Can short-circuit: know that node1.token is None
            return False

        # Both are None - how do we compare?
        return True

    FILTER_CLASSES = {
        'ident':    IdentifierNode,
        'int':      IntConstantNode,
        'float':    FloatConstantNode,
        'num':      NumConstantNode,
        'string':   StringNode,
        'const':    ConstantNode,
        'op':       OperatorNode,
        'list':     BlockNode,
        'array':    BlockNode,
        'block':    BlockNode,
    }

    def binding_node_matches(self, macro_node, ast_node):
        # No filters == immediate match
        if len(macro_node.filters) == 0:
            return True

        # We check each of the filters on the macro to see if they match.
        for flt in macro_node.filters:
            klass = self.FILTER_CLASSES[flt]
            if not isinstance(ast_node, klass):
                continue

            # If the class is a block, need to check the type too.
            if ty == 'list' and ast_node.type != '(':
                continue
            if ty == 'array' and ast_node.type != '[':
                continue
            if ty == 'block' and ast_node.type != '{':
                continue

            return True

        # Had some filters, but didn't match.
        return False

    def compare_matches(self, block, node, bindings=None):
        # To determine if the given match does in fact match the given node, we
        # compare each node with each other, including special logic for
        # macro nodes.

        if bindings is None:
            bindings = {}

        ast_node = node
        for match_node in block.children:
            if ast_node is None:
                return False, None

            # If the block node is a macro binding node - i.e. one of the
            # special $(x) variables - then it's equal if filters match.
            if isinstance(match_node, MacroBindingNode):
                if self.binding_node_matches(match_node, ast_node):
                    bindings[match_node.name] = ast_node
                    ast_node = ast_node.next
                    continue
                else:
                    return False, None

            # Command nodes should never be in the 'match' block.
            elif isinstance(match_node, MacroCommandNode):
                raise MacroError("Found macro command in 'match' block - this "
                                 "should never happen")

            # Compare the current block node and AST node.
            # Note: we check this before comparing blocks so that we catch
            # cases where both nodes are BlockNodes, but they are of different
            # types.
            if not self.are_nodes_equal(match_node, ast_node):
                return False, None

            # If they're both blocks, need to recursively check that they
            # match.
            if (isinstance(match_node, BlockNode) and
                isinstance(ast_node, BlockNode)):
                if not self.compare_matches(match_node, ast_node, bindings):
                    return False, None

            ast_node = ast_node.next

        return True, bindings

    def visit_IdentifierNode(self, node):
        if self.strip > 0:
            self.strip -= 1
            return None

        found = False
        bindings = {}
        block = None

        # Check if this is a macro identifier, and, if so, whether it's visible
        # from this location.
        for name, all_blocks in self.macros:
            if node.identifier != name:
                continue

            log.debug("Found macro invocation: %s", name)

            # To determine visibility, we need the 'macro parent' node - i.e.
            # the node in which the macro and macro's block are found.  Also,
            # the contents of 'all_blocks' are each of the 'case' blocks that
            # have been parsed, as follows:
            #   macro foo {
            #       case bar {
            #           match { baz }
            #       }
            #   }
            #
            # Thus, since all_blocks contains the curly-braced block from each
            # 'case' block, the match block's grandparent is the macro node.
            macro_node = all_blocks[0]['match'].parent.parent

            # Actually check visibility.
            if not is_macro_visible_from(macro_node, node):
                log.debug("Macro is not visible")
                continue

            # The invocation is correct.  See if one of the block's match
            # templates matches.
            for block in all_blocks:
                match = block['match']

                # NOTE: node.next since 'node' currently points to the macro
                # invocation.
                ok, bindings = self.compare_matches(block['match'], node.next)
                if ok:
                    found = True
                    break

            if found:
                break

        # If we didn't match, this isn't a 'real' invocation, and we do
        # nothing.
        if not found:
            return node

        # We've made changes.
        self.changed = True

        # We need to replace the macro with the template and then strip the
        # next 'n' nodes from the AST.
        self.strip = len(block['match'].children)

        # Clone the template AST.
        template_clone = clone_ast(block['template'])

        # Walk the template, filling in bindings.
        replacer = TemplateReplacer(bindings)
        replacer.visit(template_clone)

        # Set the 'parent' of each of the top-level template nodes to this
        # node's parents.
        for child in template_clone.children:
            child.parent = node.parent

        if 'toplevel' in block:
            toplevel_clone = clone_ast(block['toplevel'])
            replacer.visit(toplevel_clone)
            self.toplevels.append(toplevel_clone.children)

        # Return all the new tokens to replace with.  Note that we return the
        # template's children, since we don't want to include the overall
        # block.
        return template_clone.children

    def generic_visit(self, node):
        if self.strip > 0:
            self.strip -= 1
            return None

        return NodeTransformer.generic_visit(self, node)


def print_macros(macros, out=sys.stdout):
    for name, all_blocks in macros:
        print('=' * 50, file=out)
        print(name, file=out)
        print('=' * 50, file=out)
        for i, blocks in enumerate(all_blocks):
            print('Case %d' % (i + 1,), file=out)
            print('-' * 30, file=out)

            for ty, block in blocks.items():
                print(ty, file=out)
                print_ast(block, file=out)
                print('', file=out)


def process_macros(ast):
    ast, macros = strip_macros(ast)
    parsed = parse_macros(macros)

    app = MacroApplier(parsed)

    # Now, we walk the regular AST looking for cases where the templates above
    # will match.  We do this until we haven't made any changes - i.e. all
    # nested macros are applied - or until we hit a defined recursion limit of
    # 100.
    for i in range(100):
        app.reset()
        app.visit(ast)
        if not app.changed:
            break
    else:
        # In Python, 'else' statements on for loops execute if they terminated
        # due to hitting the end of the iterator, instead of a 'break'.
        raise MacroError("Hit recursion limit while expanding macros")

    # Place the toplevels.  Note that, since we're repeatedly prepending, we
    # reverse the list before inserting so the order remains the same.
    for toplevel in reversed(app.toplevels):
        ast.children[0:0] = toplevel

    return ast


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


def print_to_c(tokens, out=sys.stdout):
    # Simple pretty-printing rules:
    #   - Print the token
    #   - If the next token is not one of: ,);
    #     And if the current token is not: (
    #     Then print a space.
    #   - Increment brace / block depth
    #   - If we're at the root level and the next token is not a '}', then
    #     print a newline (to seperate top-level items).
    #   - If the current token is one of: ;{}
    #     And if we're not in brackets, then print a newline and the indent
    #     to reach the proper indentation level.
    #   - Finally, after doing this for all tokens, strip trailing whitespace.

    block_depth = 0
    brace_depth = 0
    output = []
    add = output.append

    for i, tok in enumerate(tokens):
        if i < len(tokens) - 1:
            next_tok = tokens[i + 1]
        else:
            next_tok = None

        add(tok.value)

        if ((next_tok is not None and
             next_tok.value not in [',', ')', ';']) and
            (tok.value != '(')):
            add(' ')

        if tok.value == '(':
            brace_depth += 1
        elif tok.value == ')':
            brace_depth -= 1

        if next_tok is not None and next_tok.value == '}':
            block_depth -= 1

        if tok.value in ['{', '}', ';']:
            if tok.value == '{':
                block_depth += 1

            if brace_depth == 0:
                if (((next_tok is None) or
                     (next_tok is not None and next_tok.value != '}')) and
                    block_depth == 0):
                    add('\n')
                add('\n' + ('    ' * block_depth))

    final = ''.join(output).strip()
    print(final, file=out)


###############################################################################
## USER INTERFACE


def do_expand(args):
    fname, contents = get_file(args)
    output = get_output(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)

    try:
        process_macros(ast)
    except MacroError as e:
        print("Error processing macros: %s" % (e,))

    tokens = to_tokens(ast)
    print_to_c(tokens, output)


def do_print_ast(args):
    fname, contents = get_file(args)
    output = get_output(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)

    if args.strip_macros:
        strip_macros(ast)

    print_ast(ast, output)


def do_print_macros(args):
    fname, contents = get_file(args)
    output = get_output(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    tokens = list(lexer.tokens())

    ast = make_ast(tokens)
    ast, macros = strip_macros(ast)

    parsed = parse_macros(macros)
    print_macros(parsed, output)


def do_lex(args):
    fname, contents = get_file(args)
    output = get_output(args)

    lexer = build_c_lexer()
    lexer.input(contents)
    for t in lexer.tokens():
        print("%s\t%d\t%d\t%s" % (
            t.type,
            t.line,
            t.col,
            t.value,
        ), file=output)


def get_file(args):
    fname = args.file

    if '-' == fname:
        fname = "<stdin>"
        text = sys.stdin.read()
    else:
        with open(fname, 'r') as f:
            text = f.read()

    return fname, text


def get_output(args):
    fname = args.output

    if '-' == fname:
        f = sys.stdout
    else:
        f = open(fname, 'w')

    return f


def main():
    parser = argparse.ArgumentParser(description='Macro processor for C')
    subparsers = parser.add_subparsers()

    # Global options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('file', type=str, help='file to process')
    parent_parser.add_argument('--quiet', '-q', action='store_true',
                               help='be more quiet')
    parent_parser.add_argument('--verbose', '-v', action='store_true',
                               help='be more verbose')
    parent_parser.add_argument('--output', '-o', default='-',
                               help='where to write output to')
    # TODO: add preprocessing step, argument to control preprocessor, and
    #       argument(s?) to pass flags to preprocessor

    # --------------------------------------------------
    expand_parser = subparsers.add_parser(
        'expand',
        parents=[parent_parser], help='expand macros in the input')
    expand_parser.set_defaults(func=do_expand)

    # --------------------------------------------------
    lex_parser = subparsers.add_parser(
        'lex',
        parents=[parent_parser],
        help='lex the input file, and print tokens to stdout')
    lex_parser.set_defaults(func=do_lex)

    # --------------------------------------------------
    print_ast_parser = subparsers.add_parser(
        'print_ast',
        parents=[parent_parser],
        help='build and print the AST')
    print_ast_parser.add_argument(
        "--strip-macros",
        action="store_true", help="strip macros from the AST before printing")
    print_ast_parser.set_defaults(func=do_print_ast)

    # --------------------------------------------------
    print_macros_parser = subparsers.add_parser(
        'print_macros',
        parents=[parent_parser], help='print all found macros')
    print_macros_parser.set_defaults(func=do_print_macros)

    # Parse arguments.
    args = parser.parse_args()
    if not 'func' in args:
        parser.print_help()
        return

    # Set up logger.
    if args.quiet:
        level = logging.WARN
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='[%(levelname)1.1s %(asctime)s ' \
                               '%(module)s:%(lineno)d] %(message)s',
                        level=level)
    log.debug("Starting application...")

    # Call the app.
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
