from dataclasses import dataclass
from typing import Any
from lexer import TokenType


class ParserError(Exception):
    pass


@dataclass
class ASTNode:
    token_type: str
    value: Any = None
    line: int | None = None
    column: int | None = None


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def make_node(self, token_type, value=None, token=None, anchor=None):
        if token is not None:
            return ASTNode(token_type, value, token.line, token.column)

        if anchor is not None:
            return ASTNode(token_type, value, anchor.line, anchor.column)

        current = self.current()
        return ASTNode(token_type, value, current.line, current.column)

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]

    def peek_type(self):
        return self.current().type

    def advance(self):
        tok = self.current()
        self.pos += 1
        return tok

    def check(self, token_type):
        return self.peek_type() == token_type

    def match(self, token_type):
        if self.check(token_type):
            return self.advance()
        return None

    def expect(self, token_type, message=None):
        tok = self.current()
        if tok.type != token_type:
            expected = (
                message
                or f"Expected {token_type}, got {tok.type} at line {tok.line}, column {tok.column}"
            )
            raise ParserError(expected)
        return self.advance()

    def error(self, nonterminal):
        tok = self.current()
        raise ParserError(
            f"Syntax error while parsing {nonterminal}: "
            f"unexpected token {tok.type} ('{tok.lexeme}') "
            f"at line {tok.line}, column {tok.column}"
        )

    def is_type_token(self):
        return self.peek_type() in {
            TokenType.STRING_KEYWORD,
            TokenType.INT_KEYWORD,
            TokenType.BOOL_KEYWORD,
            TokenType.LIST_KEYWORD,
        }

    def starts_statement(self):
        return self.peek_type() in {
            TokenType.STRING_KEYWORD,
            TokenType.INT_KEYWORD,
            TokenType.BOOL_KEYWORD,
            TokenType.LIST_KEYWORD,
            TokenType.ID,
            TokenType.FOR,
            TokenType.IF,
        }

    def starts_expression(self):
        return self.peek_type() in {
            TokenType.MINUS,
            TokenType.INT_LIT,
            TokenType.STRING_LIT,
            TokenType.BOOL_LIT,
            TokenType.ID,
            TokenType.LEFT_PAREN,
            TokenType.RUN,
            TokenType.LEFT_BRACKET,
        }

    def starts_action_argument(self):
        return self.peek_type() in {
            TokenType.STRING_LIT,
            TokenType.INT_LIT,
            TokenType.BOOL_LIT,
            TokenType.ID,
        }

    def parse(self):
        node = self.parse_program()
        self.expect(TokenType.EOF, "Expected end of input")
        return node

    def parse_program(self):
        if self.peek_type() in {TokenType.AGENT, TokenType.SYSTEM}:
            start_tok = self.current()
            agent_list = self.parse_agent_list()
            system_body = self.parse_system_body()
            return self.make_node(
                "Program",
                {
                    "agents": agent_list,
                    "system": system_body,
                },
                token=start_tok,
            )
        self.error("PROGRAM")

    def parse_agent_list(self):
        if self.check(TokenType.AGENT):
            agents = []
            agents.append(self.parse_agent_def())
            agents.extend(self.parse_agent_list())
            return agents
        elif self.check(TokenType.SYSTEM):
            return []

        self.error("AGENT_LIST")

    def parse_agent_def(self):
        if self.check(TokenType.AGENT):
            start_tok = self.expect(TokenType.AGENT)
            name = self.expect(TokenType.ID)
            self.expect(TokenType.LEFT_BRACE)
            body = self.parse_agent_body()
            self.expect(TokenType.RIGHT_BRACE)

            return self.make_node(
                "AgentDef",
                {
                    "name": name.lexeme,
                    "body": body,
                },
                token=start_tok,
            )
        self.error("AGENT_DEF")

    def parse_agent_body(self):
        if self.peek_type() in {TokenType.TOOL, TokenType.TASK, TokenType.RIGHT_BRACE}:
            start_tok = self.current()
            tools = self.parse_tools()
            tasks = self.parse_tasks_declarations()
            return self.make_node(
                "AgentBody",
                {
                    "tools": tools,
                    "tasks": tasks,
                },
                token=start_tok,
            )
        self.error("AGENT_BODY")

    def parse_tools(self):
        if self.check(TokenType.TOOL):
            tools = []
            tools.append(self.parse_tool_declaration())
            tools.extend(self.parse_tools())
            return tools
        elif self.peek_type() in {TokenType.TASK, TokenType.RIGHT_BRACE}:
            return []
        self.error("TOOLS")

    def parse_tool_declaration(self):
        if self.check(TokenType.TOOL):
            start_tok = self.expect(TokenType.TOOL)
            name = self.expect(TokenType.ID)
            return self.make_node("ToolDeclaration", name.lexeme, token=start_tok)
        self.error("TOOL_DECLARATION")

    def parse_tasks_declarations(self):
        if self.check(TokenType.TASK):
            tasks = []
            tasks.append(self.parse_task_declaration())
            tasks.extend(self.parse_tasks_declarations())
            return tasks
        elif self.check(TokenType.RIGHT_BRACE):
            return []
        self.error("TASKS_DECLARATIONS")

    def parse_type(self):
        if self.check(TokenType.STRING_KEYWORD):
            tok = self.expect(TokenType.STRING_KEYWORD)
            return self.make_node("Type", "string", token=tok)
        elif self.check(TokenType.INT_KEYWORD):
            tok = self.expect(TokenType.INT_KEYWORD)
            return self.make_node("Type", "int", token=tok)
        elif self.check(TokenType.LIST_KEYWORD):
            tok = self.expect(TokenType.LIST_KEYWORD)
            return self.make_node("Type", "list", token=tok)
        elif self.check(TokenType.BOOL_KEYWORD):
            tok = self.expect(TokenType.BOOL_KEYWORD)
            return self.make_node("Type", "bool", token=tok)
        self.error("TYPE")

    def parse_task_declaration(self):
        if self.check(TokenType.TASK):
            start_tok = self.expect(TokenType.TASK)
            task_name = self.expect(TokenType.ID)
            self.expect(TokenType.LEFT_PAREN)
            parameters = self.parse_task_arguments()
            self.expect(TokenType.RIGHT_PAREN)
            self.expect(TokenType.ARROW)
            return_type = self.parse_task_return_type()
            self.expect(TokenType.LEFT_BRACE)
            actions = self.parse_actions()
            self.expect(TokenType.RIGHT_BRACE)

            return self.make_node(
                "TaskDeclaration",
                {
                    "name": task_name.lexeme,
                    "parameters": parameters,
                    "return_type": return_type,
                    "actions": actions,
                },
                token=start_tok,
            )

        self.error("TASK_DECLARATION")

    def parse_task_arguments(self):
        if self.is_type_token():
            first_arg = self.parse_task_argument()
            rest_args = self.parse_task_arguments_tail()
            return [first_arg] + rest_args

        elif self.check(TokenType.RIGHT_PAREN):
            return []

        self.error("TASK_ARGUMENTS")

    def parse_task_argument(self):
        if self.is_type_token():
            start_tok = self.current()
            arg_type = self.parse_type()
            arg_name = self.expect(TokenType.ID)

            return self.make_node(
                "TaskArgument",
                {
                    "type": arg_type,
                    "name": arg_name.lexeme,
                },
                token=start_tok,
            )

        self.error("TASK_ARGUMENT")

    def parse_task_arguments_tail(self):
        if self.check(TokenType.COMMA):
            self.expect(TokenType.COMMA)
            next_arg = self.parse_task_argument()
            rest_args = self.parse_task_arguments_tail()
            return [next_arg] + rest_args

        elif self.check(TokenType.RIGHT_PAREN):
            return []

        self.error("TASK_ARGUMENTS_TAIL")

    def parse_task_return_type(self):
        if self.is_type_token():
            start_tok = self.current()
            return_type = self.parse_type()
            return_name = self.expect(TokenType.ID)

            return self.make_node(
                "TaskReturnType",
                {
                    "type": return_type,
                    "name": return_name.lexeme,
                },
                token=start_tok,
            )

        self.error("TASK_RETURN_TYPE")

    def parse_actions(self):
        if self.check(TokenType.ACTION):
            first_action = self.parse_action()
            rest_actions = self.parse_actions()
            return [first_action] + rest_actions

        elif self.check(TokenType.RIGHT_BRACE):
            return []

        self.error("ACTIONS")

    def parse_action(self):
        if self.check(TokenType.ACTION):
            start_tok = self.expect(TokenType.ACTION)
            self.expect(TokenType.COLON)
            action_name = self.expect(TokenType.ID)
            self.expect(TokenType.LEFT_PAREN)
            arguments = self.parse_action_arguments()
            self.expect(TokenType.RIGHT_PAREN)

            return self.make_node(
                "Action",
                {
                    "name": action_name.lexeme,
                    "arguments": arguments,
                },
                token=start_tok,
            )

        self.error("ACTION")

    def parse_action_arguments(self):
        if self.starts_action_argument():
            first_arg = self.parse_action_argument()
            rest_args = self.parse_action_arguments_tail()
            return [first_arg] + rest_args

        elif self.check(TokenType.RIGHT_PAREN):
            return []

        self.error("ACTION_ARGUMENTS")

    def parse_action_argument(self):
        if self.check(TokenType.STRING_LIT):
            tok = self.expect(TokenType.STRING_LIT)
            return self.make_node(
                "ActionArgument",
                {
                    "kind": "string",
                    "value": tok.lexeme,
                },
                token=tok,
            )

        elif self.check(TokenType.INT_LIT):
            tok = self.expect(TokenType.INT_LIT)
            return self.make_node(
                "ActionArgument",
                {
                    "kind": "int",
                    "value": tok.lexeme,
                },
                token=tok,
            )

        elif self.check(TokenType.BOOL_LIT):
            tok = self.expect(TokenType.BOOL_LIT)
            return self.make_node(
                "ActionArgument",
                {
                    "kind": "bool",
                    "value": tok.lexeme,
                },
                token=tok,
            )

        elif self.check(TokenType.ID):
            tok = self.expect(TokenType.ID)
            return self.make_node(
                "ActionArgument",
                {
                    "kind": "id",
                    "value": tok.lexeme,
                },
                token=tok,
            )

        self.error("ACTION_ARGUMENT")

    def parse_action_arguments_tail(self):
        if self.check(TokenType.COMMA):
            self.expect(TokenType.COMMA)
            next_arg = self.parse_action_argument()
            rest_args = self.parse_action_arguments_tail()
            return [next_arg] + rest_args

        elif self.check(TokenType.RIGHT_PAREN):
            return []

        self.error("ACTION_ARGUMENTS_TAIL")

    def parse_system_body(self):
        if self.check(TokenType.SYSTEM):
            start_tok = self.expect(TokenType.SYSTEM)
            self.expect(TokenType.LEFT_BRACE)
            statements = self.parse_statements()
            self.expect(TokenType.RIGHT_BRACE)

            return self.make_node(
                "SystemBody",
                {
                    "statements": statements,
                },
                token=start_tok,
            )

        self.error("SYSTEM_BODY")

    def parse_statements(self):
        if self.starts_statement():
            first_stmt = self.parse_statement()
            rest_stmts = self.parse_statements()
            return [first_stmt] + rest_stmts

        elif self.check(TokenType.RIGHT_BRACE):
            return []

        self.error("STATEMENTS")

    def parse_statement(self):
        if self.is_type_token():
            return self.parse_variable_declaration()

        elif self.check(TokenType.ID):
            return self.parse_variable_assignment()

        elif self.check(TokenType.FOR):
            return self.parse_for_statement()

        elif self.check(TokenType.IF):
            return self.parse_if_statement()

        self.error("STATEMENT")

    def parse_variable_declaration(self):
        if self.is_type_token():
            start_tok = self.current()
            var_type = self.parse_type()
            var_name = self.expect(TokenType.ID)
            self.expect(TokenType.ASSIGN)
            expr = self.parse_expression()

            return self.make_node(
                "VariableDeclaration",
                {
                    "type": var_type,
                    "name": var_name.lexeme,
                    "value": expr,
                },
                token=start_tok,
            )

        self.error("VARIABLE_DECLARATION")

    def parse_variable_assignment(self):
        if self.check(TokenType.ID):
            start_tok = self.expect(TokenType.ID)
            self.expect(TokenType.ASSIGN)
            expr = self.parse_expression()

            return self.make_node(
                "VariableAssignment",
                {
                    "name": start_tok.lexeme,
                    "value": expr,
                },
                token=start_tok,
            )

        self.error("VARIABLE_ASSIGNMENT")

    def parse_for_statement(self):
        if self.check(TokenType.FOR):
            start_tok = self.expect(TokenType.FOR)
            loop_var = self.expect(TokenType.ID)
            self.expect(TokenType.IN)
            iterable_name = self.expect(TokenType.ID)
            self.expect(TokenType.LEFT_BRACE)
            body = self.parse_statements()
            self.expect(TokenType.RIGHT_BRACE)

            return self.make_node(
                "ForStatement",
                {
                    "iterator": loop_var.lexeme,
                    "iterable": iterable_name.lexeme,
                    "body": body,
                },
                token=start_tok,
            )

        self.error("FOR_STATEMENT")

    def parse_if_statement(self):
        if self.check(TokenType.IF):
            start_tok = self.expect(TokenType.IF)
            condition = self.parse_expression()
            self.expect(TokenType.LEFT_BRACE)
            body = self.parse_statements()
            self.expect(TokenType.RIGHT_BRACE)

            return self.make_node(
                "IfStatement",
                {
                    "condition": condition,
                    "body": body,
                },
                token=start_tok,
            )

        self.error("IF_STATEMENT")

    def is_rel_op(self):
        return self.peek_type() in {
            TokenType.EQUAL_EQUAL,
            TokenType.LESS_EQUAL,
            TokenType.LESS,
            TokenType.GREATER_EQUAL,
            TokenType.GREATER,
            TokenType.NOT_EQUAL,
        }

    def in_expression_follow(self):
        return self.peek_type() in {
            TokenType.RIGHT_PAREN,
            TokenType.COMMA,
            TokenType.RIGHT_BRACKET,
            TokenType.LEFT_BRACE,
            TokenType.STRING_KEYWORD,
            TokenType.INT_KEYWORD,
            TokenType.LIST_KEYWORD,
            TokenType.BOOL_KEYWORD,
            TokenType.ID,
            TokenType.FOR,
            TokenType.IF,
            TokenType.RIGHT_BRACE,
            TokenType.EOF,
        }

    def in_add_expr_follow(self):
        return self.peek_type() in {
            TokenType.EQUAL_EQUAL,
            TokenType.LESS_EQUAL,
            TokenType.LESS,
            TokenType.GREATER_EQUAL,
            TokenType.GREATER,
            TokenType.NOT_EQUAL,
            TokenType.RIGHT_PAREN,
            TokenType.COMMA,
            TokenType.RIGHT_BRACKET,
            TokenType.LEFT_BRACE,
            TokenType.STRING_KEYWORD,
            TokenType.INT_KEYWORD,
            TokenType.LIST_KEYWORD,
            TokenType.BOOL_KEYWORD,
            TokenType.ID,
            TokenType.FOR,
            TokenType.IF,
            TokenType.RIGHT_BRACE,
            TokenType.EOF,
        }

    def in_term_follow(self):
        return self.peek_type() in {
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.EQUAL_EQUAL,
            TokenType.LESS_EQUAL,
            TokenType.LESS,
            TokenType.GREATER_EQUAL,
            TokenType.GREATER,
            TokenType.NOT_EQUAL,
            TokenType.RIGHT_PAREN,
            TokenType.COMMA,
            TokenType.RIGHT_BRACKET,
            TokenType.LEFT_BRACE,
            TokenType.STRING_KEYWORD,
            TokenType.INT_KEYWORD,
            TokenType.LIST_KEYWORD,
            TokenType.BOOL_KEYWORD,
            TokenType.ID,
            TokenType.FOR,
            TokenType.IF,
            TokenType.RIGHT_BRACE,
            TokenType.EOF,
        }

    def parse_expression(self):
        if self.starts_expression():
            left = self.parse_add_expr()
            return self.parse_rel_tail(left)

        self.error("EXPRESSION")

    def parse_rel_tail(self, left):
        if self.is_rel_op():
            op_tok = self.current()
            op = self.parse_operation()
            right = self.parse_add_expr()
            node = self.make_node(
                "BinaryOp",
                {
                    "op": op,
                    "left": left,
                    "right": right,
                },
                token=op_tok,
            )
            return node

        elif self.in_expression_follow():
            return left

        self.error("REL_TAIL")

    def parse_add_expr(self):
        if self.starts_expression():
            left = self.parse_term()
            return self.parse_add_tail(left)

        self.error("ADD_EXPR")

    def parse_add_tail(self, left):
        if self.check(TokenType.PLUS):
            op_tok = self.expect(TokenType.PLUS)
            right = self.parse_term()
            node = self.make_node(
                "BinaryOp",
                {
                    "op": "+",
                    "left": left,
                    "right": right,
                },
                token=op_tok,
            )
            return self.parse_add_tail(node)

        elif self.check(TokenType.MINUS):
            op_tok = self.expect(TokenType.MINUS)
            right = self.parse_term()
            node = self.make_node(
                "BinaryOp",
                {
                    "op": "-",
                    "left": left,
                    "right": right,
                },
                token=op_tok,
            )
            return self.parse_add_tail(node)

        elif self.in_add_expr_follow():
            return left

        self.error("ADD_TAIL")

    def parse_term(self):
        if self.starts_expression():
            left = self.parse_factor()
            return self.parse_term_tail(left)

        self.error("TERM")

    def parse_term_tail(self, left):
        if self.check(TokenType.MULT):
            op_tok = self.expect(TokenType.MULT)
            right = self.parse_factor()
            node = self.make_node(
                "BinaryOp",
                {
                    "op": "*",
                    "left": left,
                    "right": right,
                },
                token=op_tok,
            )
            return self.parse_term_tail(node)

        elif self.check(TokenType.DIV):
            op_tok = self.expect(TokenType.DIV)
            right = self.parse_factor()
            node = self.make_node(
                "BinaryOp",
                {
                    "op": "/",
                    "left": left,
                    "right": right,
                },
                token=op_tok,
            )
            return self.parse_term_tail(node)

        elif self.in_term_follow():
            return left

        self.error("TERM_TAIL")

    def parse_factor(self):
        if self.check(TokenType.MINUS):
            op_tok = self.expect(TokenType.MINUS)
            operand = self.parse_factor()
            return self.make_node(
                "UnaryOp",
                {
                    "op": "-",
                    "operand": operand,
                },
                token=op_tok,
            )

        elif self.check(TokenType.INT_LIT):
            tok = self.expect(TokenType.INT_LIT)
            return self.make_node("IntLiteral", tok.lexeme, token=tok)

        elif self.check(TokenType.STRING_LIT):
            tok = self.expect(TokenType.STRING_LIT)
            return self.make_node("StringLiteral", tok.lexeme, token=tok)

        elif self.check(TokenType.BOOL_LIT):
            tok = self.expect(TokenType.BOOL_LIT)
            return self.make_node("BoolLiteral", tok.lexeme, token=tok)

        elif self.check(TokenType.ID):
            tok = self.expect(TokenType.ID)
            return self.make_node("Identifier", tok.lexeme, token=tok)

        elif self.check(TokenType.LEFT_PAREN):
            self.expect(TokenType.LEFT_PAREN)
            expr = self.parse_expression()
            self.expect(TokenType.RIGHT_PAREN)
            return expr

        elif self.check(TokenType.RUN):
            return self.parse_function_call()

        elif self.check(TokenType.LEFT_BRACKET):
            return self.parse_list_declaration()

        self.error("FACTOR")

    def parse_function_call(self):
        if self.check(TokenType.RUN):
            start_tok = self.expect(TokenType.RUN)
            agent_name = self.expect(TokenType.ID)
            self.expect(TokenType.DOT)
            task_name = self.expect(TokenType.ID)
            self.expect(TokenType.LEFT_PAREN)
            arguments = self.parse_function_arguments()
            self.expect(TokenType.RIGHT_PAREN)

            return self.make_node(
                "FunctionCall",
                {
                    "agent": agent_name.lexeme,
                    "task": task_name.lexeme,
                    "arguments": arguments,
                },
                token=start_tok,
            )

        self.error("FUNCTION_CALL")

    def parse_function_arguments(self):
        if self.check(TokenType.ID):
            return self.parse_function_argument_list()

        elif self.check(TokenType.RIGHT_PAREN):
            return []

        self.error("FUNCTION_ARGUMENTS")

    def parse_function_argument_list(self):
        if self.check(TokenType.ID):
            first_arg = self.parse_function_argument()
            rest_args = self.parse_function_argument_tail()
            return [first_arg] + rest_args

        self.error("FUNCTION_ARGUMENT_LIST")

    def parse_function_argument(self):
        if self.check(TokenType.ID):
            tok = self.expect(TokenType.ID)
            return self.make_node("Identifier", tok.lexeme, token=tok)

        self.error("FUNCTION_ARGUMENT")

    def parse_function_argument_tail(self):
        if self.check(TokenType.COMMA):
            self.expect(TokenType.COMMA)
            next_arg = self.parse_function_argument()
            rest_args = self.parse_function_argument_tail()
            return [next_arg] + rest_args

        elif self.check(TokenType.RIGHT_PAREN):
            return []

        self.error("FUNCTION_ARGUMENT_TAIL")

    def parse_operation(self):
        if self.check(TokenType.EQUAL_EQUAL):
            self.expect(TokenType.EQUAL_EQUAL)
            return "=="

        elif self.check(TokenType.LESS_EQUAL):
            self.expect(TokenType.LESS_EQUAL)
            return "<="

        elif self.check(TokenType.LESS):
            self.expect(TokenType.LESS)
            return "<"

        elif self.check(TokenType.GREATER_EQUAL):
            self.expect(TokenType.GREATER_EQUAL)
            return ">="

        elif self.check(TokenType.GREATER):
            self.expect(TokenType.GREATER)
            return ">"

        elif self.check(TokenType.NOT_EQUAL):
            self.expect(TokenType.NOT_EQUAL)
            return "!="

        self.error("OPERATION")

    def parse_list_declaration(self):
        if self.check(TokenType.LEFT_BRACKET):
            start_tok = self.expect(TokenType.LEFT_BRACKET)
            items = self.parse_list_items()
            self.expect(TokenType.RIGHT_BRACKET)

            return self.make_node("ListLiteral", items, token=start_tok)

        self.error("LIST_DECLARATION")

    def parse_list_items(self):
        if self.starts_expression():
            first_item = self.parse_expression()
            rest_items = self.parse_list_items_tail()
            return [first_item] + rest_items

        elif self.check(TokenType.RIGHT_BRACKET):
            return []

        self.error("LIST_ITEMS")

    def parse_list_items_tail(self):
        if self.check(TokenType.COMMA):
            self.expect(TokenType.COMMA)
            next_item = self.parse_expression()
            rest_items = self.parse_list_items_tail()
            return [next_item] + rest_items

        elif self.check(TokenType.RIGHT_BRACKET):
            return []

        self.error("LIST_ITEMS_TAIL")
