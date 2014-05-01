#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import unittest
from functools import wraps
from textwrap import dedent

sys.path.insert(0, '.')
import cmacro


# Get around lack of locality.
ASSERT_COUNTER = [0]


def counting_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ASSERT_COUNTER[0] += 1
        return func(*args, **kwargs)

    return wrapper


class BaseTestCase(unittest.TestCase):
    def __getattribute__(self, attr_name):
        obj = super(BaseTestCase, self).__getattribute__(attr_name)

        # Only count 'assert' functions that come from the base class.
        if (hasattr(obj, "__call__") and attr_name.startswith('assert') and
            hasattr(unittest.TestCase, attr_name)):
            return counting_wrapper(obj)
        else:
            return obj


class LexingTest(BaseTestCase):
    def load_passing_testcases(self):
        pth = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "tests",
            "passing"
        )
        if not (os.path.exists(pth) and os.path.isdir(pth)):
            return

        for fname in os.listdir(pth):
            base, ext = os.path.splitext(fname)
            if ext == '.c':
                with open(os.path.join(pth, fname), 'r') as f:
                    code = f.read()
                with open(os.path.join(pth, base + '.results'), 'r') as f:
                    results = f.read()

                # Split results.  Format is:
                #   TYPE\tLINE\tCOL\tVALUE...
                lines = results.splitlines()
                tuples = [x.split('\t', 4) for x in lines
                          if not x.startswith('#') and len(x.strip()) > 0]
                tokens = [
                    cmacro.Token(x[0], x[3], int(x[1]), int(x[2]))
                    for x in tuples
                ]

                yield fname, code, tokens

    def load_failing_testcases(self):
        pth = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "tests",
            "failing"
        )
        if not (os.path.exists(pth) and os.path.isdir(pth)):
            return

        for f in os.listdir(pth):
            base, ext = os.path.splitext(f)
            if ext == '.c':
                with open(f, 'rb') as f:
                    code = f.read()

                # TODO: need to load results somehow!
                yield code

    def setUp(self):
        self.lexer = cmacro.build_c_lexer()

    def assertTokensEqual(self, t1, t2):
        self.assertEqual(t1.type, t2.type)
        self.assertEqual(t1.value, t2.value)
        self.assertEqual(t1.line, t2.line)
        self.assertEqual(t1.col, t2.col)

    def test_passing(self):
        for pth, code, results in self.load_passing_testcases():
            with self.subTest(file=pth):
                # Lex the input file.
                self.lexer.input(code)
                tokens = list(self.lexer.tokens())

                # Compare the two sets of tokens.
                self.assertEqual(len(tokens), len(results))
                for ours, res in zip(tokens, results):
                    self.assertTokensEqual(ours, res)

    def test_failing(self):
        for pth, code, results in self.load_failing_testcases():
            with self.subTest(file=pth):
                # Lex the input file.
                self.lexer.input(code)
                tokens = list(self.lexer.tokens())

                # TODO: how do we compare for errors?


class TestMakeAst(BaseTestCase):
    def test_unbalanced(self):
        tokens = [
            cmacro.Token('OPERATOR', '{', -1, -1),
            cmacro.Token('OPERATOR', '(', -1, -1),
            cmacro.Token('IDENTIFIER', 'foo', -1, -1),
            cmacro.Token('OPERATOR', ')', -1, -1),
            # missing closing '}' deliberately
        ]

        with self.assertRaises(ValueError):
            cmacro.make_ast(tokens)

    def test_invalid_token(self):
        with self.assertRaises(RuntimeError):
            cmacro.make_ast([
                cmacro.Token('UNKNOWN', 'bad', -1, -1),
            ])


class TestMacroStripper(BaseTestCase):
    def run_test(self, code):
        lexer = cmacro.build_c_lexer()
        lexer.input(dedent(code).strip())
        tokens = list(lexer.tokens())

        ast = cmacro.make_ast(tokens)
        new_ast, macros = cmacro.strip_macros(ast)

        return new_ast, macros

    def test_strip(self):
        code = """
            macro foo {
                doesntmatter
            }
        """

        new_ast, macros = self.run_test(code)
        self.assertEqual(len(new_ast.children), 0)
        self.assertEqual(len(macros), 1)
        self.assertEqual(macros[0][0], 'foo')

    def test_invalid_macro_block(self):
        code = "macro"

        with self.assertRaises(cmacro.MacroError) as cm:
            self.run_test(code)

        exc = cm.exception
        self.assertTrue(exc.msg.startswith("No macro name found after"))
        self.assertEqual(exc.line, 1)

    def test_invalid_macro_name(self):
        code = """
        macro 1234 {
            case {
                match {
                    bar
                }
                template {
                }
            }
        }
        """

        with self.assertRaises(cmacro.MacroError) as cm:
            self.run_test(code)

        exc = cm.exception
        self.assertTrue(exc.msg.startswith("Expected token after 'macro' to"))
        self.assertEqual(exc.line, 1)

    def test_missing_macro_block(self):
        code = "macro foo"

        with self.assertRaises(cmacro.MacroError) as cm:
            self.run_test(code)

        exc = cm.exception
        self.assertTrue(exc.msg.startswith("No block found after macro name"))
        self.assertEqual(exc.line, 1)

    def test_invalid_macro_block(self):
        code = "macro foo 123"

        with self.assertRaises(cmacro.MacroError) as cm:
            self.run_test(code)

        exc = cm.exception
        self.assertTrue(exc.msg.startswith("Expected token after name to be "
                                           "a block"))
        self.assertEqual(exc.line, 1)

    def test_invalid_macro_block_type(self):
        code = "macro foo ( )"

        with self.assertRaises(cmacro.MacroError) as cm:
            self.run_test(code)

        exc = cm.exception
        self.assertTrue(exc.msg.startswith("Expected token after name to be "
                                           "a block"))
        self.assertEqual(exc.line, 1)


class TestMacroVisitor(BaseTestCase):
    pass


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        print("ASSERTS: %d" % (ASSERT_COUNTER[0],), file=sys.stderr)
