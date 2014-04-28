#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import unittest

sys.path.insert(0, '.')
import cmacro


class LexingTest(unittest.TestCase):
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
                tuples = [x.split('\t', 4) for x in lines]
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
        """ Test all files in ./tests/passing/*.c to ensure that we
            lex them correctly.
        """
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
        """ Test all files in ./tests/failing/*.c to ensure that we
            get a correct error.
        """
        for pth, code, results in self.load_failing_testcases():
            with self.subTest(file=pth):
                # Lex the input file.
                self.lexer.input(code)
                tokens = list(self.lexer.tokens())

                # TODO: how do we compare for errors?


class ExpansionTest(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
