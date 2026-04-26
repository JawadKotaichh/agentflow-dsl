from lexer import Lexer
from parser import Parser
from print_parse_tree import print_parse_tree
from semantic import SemanticAnalyzer


source = """
    agent Researcher {
        tool web_search
        tool llm
        task gather(string topic) -> string data {
            action: web_search(topic)
            action: llm("summarize results")
        }
    }
    agent Analyzer {
        tool llm
        task sentiment(string text) -> string result {
            action: llm("detect sentiment")
        }
    }
    system {
        list topics = ["AI","Robotics","Security"]
        int i = 0
        bool negative_found = false
        for t in topics {
            string data = run Researcher.gather(t)
            string sentiment = run Analyzer.sentiment(data)
            if sentiment == "negative" {
                negative_found = true
            }
            i = i + 1
        }
    }
"""
tokens = Lexer(source).tokenize()
ast = Parser(tokens).parse()
print_parse_tree(ast)

analysis = SemanticAnalyzer().analyze(ast)
print("\nSemantic analysis: OK")
print(analysis.symbol_tables())
