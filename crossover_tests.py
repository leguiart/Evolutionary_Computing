import unittest
from test_genetic_algorithm import TestSinglePointCrossover

class TestCrossoverMethods(unittest.TestCase):

    def setUp(self):
        self.ga = TestSinglePointCrossover([[0.25410149, 0.71410111, 0.31915886, 0.45725239]], [[0.25410149, 0.71410111, 0.31915886, 0.45725239]])

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_single_point(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()