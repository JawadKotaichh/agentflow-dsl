from __future__ import annotations
from dataclasses import dataclass, field
from typing import NoReturn
from parser import ASTNode


class SemanticError(Exception):
    pass


@dataclass(frozen=True)
class SemanticType:
    name: str
    element_type: "SemanticType | None" = None

    def describe(self) -> str:
        if self.name != "list":
            return self.name

        if self.element_type is None or self.element_type.name == "unknown":
            return "list"

        return f"list[{self.element_type.describe()}]"


UNKNOWN_TYPE = SemanticType("unknown")
STRING_TYPE = SemanticType("string")
INT_TYPE = SemanticType("int")
BOOL_TYPE = SemanticType("bool")
LIST_TYPE = SemanticType("list")


@dataclass
class VariableSymbol:
    name: str
    declared_type: SemanticType
    current_type: SemanticType
    node: ASTNode


@dataclass
class TaskSymbol:
    name: str
    parameters: list[VariableSymbol]
    return_type: SemanticType
    return_name: str
    node: ASTNode


@dataclass
class AgentSymbol:
    name: str
    tools: dict[str, ASTNode] = field(default_factory=dict)
    tasks: dict[str, TaskSymbol] = field(default_factory=dict)
    node: ASTNode | None = None


@dataclass
class Scope:
    name: str
    parent: "Scope | None" = None
    symbols: dict[str, VariableSymbol] = field(default_factory=dict)

    def define(self, symbol: VariableSymbol) -> None:
        self.symbols[symbol.name] = symbol

    def lookup_local(self, name: str) -> VariableSymbol | None:
        return self.symbols.get(name)

    def lookup(self, name: str) -> VariableSymbol | None:
        scope: Scope | None = self
        while scope is not None:
            symbol = scope.lookup_local(name)
            if symbol is not None:
                return symbol
            scope = scope.parent
        return None

    def snapshot(self) -> dict[str, str]:
        return {
            name: symbol.current_type.describe()
            for name, symbol in sorted(self.symbols.items())
        }

    def declared_snapshot(self) -> dict[str, str]:
        return {
            name: symbol.declared_type.describe()
            for name, symbol in sorted(self.symbols.items())
        }


class SemanticAnalyzer:
    def __init__(self) -> None:
        self.agents: dict[str, AgentSymbol] = {}
        self.system_scope: Scope | None = None

    def analyze(self, ast: ASTNode) -> "SemanticAnalyzer":
        if ast.token_type != "Program":
            self._error(ast, "Expected Program node at the root of the AST")

        agent_nodes = ast.value["agents"]
        for agent_node in agent_nodes:
            self._build_agent_symbol_table(agent_node)

        for agent_node in agent_nodes:
            self._check_agent_semantics(agent_node)

        self.system_scope = Scope("system")
        self._check_system_semantics(ast.value["system"], self.system_scope)
        return self

    def symbol_tables(self) -> dict[str, object]:
        return {
            "agents": {
                agent_name: {
                    "tools": sorted(agent.tools),
                    "tasks": {
                        task_name: {
                            "parameters": [
                                {
                                    "name": param.name,
                                    "type": param.declared_type.describe(),
                                }
                                for param in task.parameters
                            ],
                            "return_type": task.return_type.describe(),
                            "return_name": task.return_name,
                        }
                        for task_name, task in sorted(agent.tasks.items())
                    },
                }
                for agent_name, agent in sorted(self.agents.items())
            },
            "system": (
                {}
                if self.system_scope is None
                else self.system_scope.declared_snapshot()
            ),
        }

    def _build_agent_symbol_table(self, agent_node: ASTNode) -> None:
        agent_name = agent_node.value["name"]
        if agent_name in self.agents:
            self._error(agent_node, f"Agent '{agent_name}' is already declared")

        agent_symbol = AgentSymbol(name=agent_name, node=agent_node)
        agent_body = agent_node.value["body"]

        for tool_node in agent_body.value["tools"]:
            tool_name = tool_node.value
            if tool_name in agent_symbol.tools:
                self._error(
                    tool_node,
                    f"Tool '{tool_name}' is already declared in agent '{agent_name}'",
                )
            agent_symbol.tools[tool_name] = tool_node

        for task_node in agent_body.value["tasks"]:
            task_name = task_node.value["name"]
            if task_name in agent_symbol.tasks:
                self._error(
                    task_node,
                    f"Task '{task_name}' is already declared in agent '{agent_name}'",
                )

            parameters = self._build_task_parameter_symbols(task_node)
            return_info = task_node.value["return_type"]
            return_type = self._type_from_node(return_info.value["type"])
            return_name = return_info.value["name"]
            if any(parameter.name == return_name for parameter in parameters):
                self._error(
                    return_info,
                    f"Return name '{return_name}' is already used as a parameter "
                    f"in task '{task_name}'",
                )
            agent_symbol.tasks[task_name] = TaskSymbol(
                name=task_name,
                parameters=parameters,
                return_type=return_type,
                return_name=return_name,
                node=task_node,
            )

        self.agents[agent_name] = agent_symbol

    def _build_task_parameter_symbols(
        self,
        task_node: ASTNode,
    ) -> list[VariableSymbol]:
        parameters: list[VariableSymbol] = []
        seen: set[str] = set()

        for parameter_node in task_node.value["parameters"]:
            parameter_name = parameter_node.value["name"]
            if parameter_name in seen:
                self._error(
                    parameter_node,
                    f"Parameter '{parameter_name}' is already declared in task "
                    f"'{task_node.value['name']}'",
                )

            declared_type = self._type_from_node(parameter_node.value["type"])
            parameters.append(
                VariableSymbol(
                    name=parameter_name,
                    declared_type=declared_type,
                    current_type=declared_type,
                    node=parameter_node,
                )
            )
            seen.add(parameter_name)

        return parameters

    def _check_agent_semantics(self, agent_node: ASTNode) -> None:
        agent_name = agent_node.value["name"]
        agent_symbol = self.agents[agent_name]

        for task_node in agent_node.value["body"].value["tasks"]:
            task_symbol = agent_symbol.tasks[task_node.value["name"]]
            self._check_task_semantics(agent_symbol, task_symbol, task_node)

    def _check_task_semantics(
        self,
        agent_symbol: AgentSymbol,
        task_symbol: TaskSymbol,
        task_node: ASTNode,
    ) -> None:
        task_scope = Scope(f"task:{agent_symbol.name}.{task_symbol.name}")

        for parameter in task_symbol.parameters:
            task_scope.define(parameter)

        for action_node in task_node.value["actions"]:
            self._check_action_semantics(agent_symbol, task_scope, action_node)

    def _check_action_semantics(
        self,
        agent_symbol: AgentSymbol,
        scope: Scope,
        action_node: ASTNode,
    ) -> None:
        action_name = action_node.value["name"]
        if action_name not in agent_symbol.tools:
            self._error(
                action_node,
                f"Tool '{action_name}' is not declared in agent '{agent_symbol.name}'",
            )

        for argument_node in action_node.value["arguments"]:
            if argument_node.value["kind"] != "id":
                continue

            argument_name = argument_node.value["value"]
            if scope.lookup(argument_name) is None:
                self._error(
                    argument_node,
                    f"Identifier '{argument_name}' is used before declaration",
                )

    def _check_system_semantics(self, system_node: ASTNode, scope: Scope) -> None:
        for statement in system_node.value["statements"]:
            self._check_statement_semantics(statement, scope)

    def _check_statement_semantics(
        self,
        statement_node: ASTNode,
        scope: Scope,
    ) -> None:
        kind = statement_node.token_type

        if kind == "VariableDeclaration":
            self._check_variable_declaration(statement_node, scope)
            return

        if kind == "VariableAssignment":
            self._check_variable_assignment(statement_node, scope)
            return

        if kind == "ForStatement":
            self._check_for_statement(statement_node, scope)
            return

        if kind == "IfStatement":
            self._check_if_statement(statement_node, scope)
            return

        self._error(statement_node, f"Unsupported statement node '{kind}'")

    def _check_variable_declaration(
        self,
        declaration_node: ASTNode,
        scope: Scope,
    ) -> None:
        variable_name = declaration_node.value["name"]
        if scope.lookup_local(variable_name) is not None:
            self._error(
                declaration_node,
                f"Variable '{variable_name}' is already declared in the same scope",
            )

        declared_type = self._type_from_node(declaration_node.value["type"])
        value_type = self._type_check_expression(
            declaration_node.value["value"],
            scope,
        )
        if not self._is_assignable(declared_type, value_type):
            self._error(
                declaration_node,
                f"Cannot assign value of type '{value_type.describe()}' to variable "
                f"'{variable_name}' of type '{declared_type.describe()}'",
            )

        scope.define(
            VariableSymbol(
                name=variable_name,
                declared_type=declared_type,
                current_type=self._merge_declared_and_value_type(
                    declared_type, value_type
                ),
                node=declaration_node,
            )
        )

    def _check_variable_assignment(
        self,
        assignment_node: ASTNode,
        scope: Scope,
    ) -> None:
        variable_name = assignment_node.value["name"]
        symbol = scope.lookup(variable_name)
        if symbol is None:
            self._error(
                assignment_node,
                f"Variable '{variable_name}' is assigned before declaration",
            )

        value_type = self._type_check_expression(assignment_node.value["value"], scope)
        if not self._is_assignable(symbol.declared_type, value_type):
            self._error(
                assignment_node,
                f"Cannot assign value of type '{value_type.describe()}' to variable "
                f"'{variable_name}' of type '{symbol.declared_type.describe()}'",
            )

        symbol.current_type = self._merge_declared_and_value_type(
            symbol.declared_type,
            value_type,
        )

    def _check_for_statement(self, for_node: ASTNode, scope: Scope) -> None:
        iterable_name = for_node.value["iterable"]
        iterable_symbol = scope.lookup(iterable_name)
        if iterable_symbol is None:
            self._error(
                for_node,
                f"Variable '{iterable_name}' is used before declaration",
            )

        iterable_type = iterable_symbol.current_type
        if iterable_type.name != "list":
            self._error(
                for_node,
                f"For-loop iterable '{iterable_name}' must have type 'list', "
                f"got '{iterable_type.describe()}'",
            )

        iterator_type = iterable_type.element_type or UNKNOWN_TYPE
        loop_scope = Scope("for", parent=scope)
        loop_scope.define(
            VariableSymbol(
                name=for_node.value["iterator"],
                declared_type=iterator_type,
                current_type=iterator_type,
                node=for_node,
            )
        )

        for statement in for_node.value["body"]:
            self._check_statement_semantics(statement, loop_scope)

    def _check_if_statement(self, if_node: ASTNode, scope: Scope) -> None:
        condition_type = self._type_check_expression(if_node.value["condition"], scope)
        if condition_type.name not in {"bool", "unknown"}:
            self._error(
                if_node,
                "If-condition must have type 'bool', "
                f"got '{condition_type.describe()}'",
            )

        if_scope = Scope("if", parent=scope)
        for statement in if_node.value["body"]:
            self._check_statement_semantics(statement, if_scope)

    def _type_check_expression(
        self,
        expression_node: ASTNode,
        scope: Scope,
    ) -> SemanticType:
        kind = expression_node.token_type

        if kind == "IntLiteral":
            return INT_TYPE

        if kind == "StringLiteral":
            return STRING_TYPE

        if kind == "BoolLiteral":
            return BOOL_TYPE

        if kind == "Identifier":
            symbol = scope.lookup(expression_node.value)
            if symbol is None:
                self._error(
                    expression_node,
                    f"Variable '{expression_node.value}' is used before declaration",
                )
            return symbol.current_type

        if kind == "UnaryOp":
            operand_type = self._type_check_expression(
                expression_node.value["operand"], scope
            )
            if operand_type.name not in {"int", "unknown"}:
                self._error(
                    expression_node,
                    "Unary '-' requires an operand of type 'int', "
                    f"got '{operand_type.describe()}'",
                )
            return INT_TYPE if operand_type.name == "int" else UNKNOWN_TYPE

        if kind == "BinaryOp":
            return self._type_check_binary_expression(expression_node, scope)

        if kind == "FunctionCall":
            return self._type_check_function_call(expression_node, scope)

        if kind == "ListLiteral":
            return self._type_check_list_literal(expression_node, scope)

        self._error(expression_node, f"Unsupported expression node '{kind}'")

    def _type_check_binary_expression(
        self,
        expression_node: ASTNode,
        scope: Scope,
    ) -> SemanticType:
        operator = expression_node.value["op"]
        left_type = self._type_check_expression(expression_node.value["left"], scope)
        right_type = self._type_check_expression(expression_node.value["right"], scope)

        if operator in {"+", "-", "*", "/"}:
            if left_type.name not in {"int", "unknown"} or right_type.name not in {
                "int",
                "unknown",
            }:
                self._error(
                    expression_node,
                    f"Operator '{operator}' requires integer operands, got "
                    f"'{left_type.describe()}' and '{right_type.describe()}'",
                )
            if "unknown" in {left_type.name, right_type.name}:
                return UNKNOWN_TYPE
            return INT_TYPE

        if operator in {"<", "<=", ">", ">="}:
            if left_type.name not in {"int", "unknown"} or right_type.name not in {
                "int",
                "unknown",
            }:
                self._error(
                    expression_node,
                    f"Operator '{operator}' requires integer operands, got "
                    f"'{left_type.describe()}' and '{right_type.describe()}'",
                )
            return BOOL_TYPE

        if operator in {"==", "!="}:
            if not self._are_comparable(left_type, right_type):
                self._error(
                    expression_node,
                    f"Cannot compare values of type '{left_type.describe()}' and "
                    f"'{right_type.describe()}'",
                )
            return BOOL_TYPE

        self._error(expression_node, f"Unsupported operator '{operator}'")

    def _type_check_function_call(
        self,
        call_node: ASTNode,
        scope: Scope,
    ) -> SemanticType:
        agent_name = call_node.value["agent"]
        task_name = call_node.value["task"]

        agent_symbol = self.agents.get(agent_name)
        if agent_symbol is None:
            self._error(call_node, f"Agent '{agent_name}' is not declared")

        task_symbol = agent_symbol.tasks.get(task_name)
        if task_symbol is None:
            self._error(
                call_node,
                f"Task '{task_name}' is not declared in agent '{agent_name}'",
            )

        arguments = call_node.value["arguments"]
        if len(arguments) != len(task_symbol.parameters):
            self._error(
                call_node,
                f"Task '{agent_name}.{task_name}' expects {len(task_symbol.parameters)} "
                f"argument(s), got {len(arguments)}",
            )

        for argument_node, parameter in zip(arguments, task_symbol.parameters):
            argument_type = self._type_check_expression(argument_node, scope)
            if not self._is_assignable(parameter.declared_type, argument_type):
                self._error(
                    argument_node,
                    f"Argument for parameter '{parameter.name}' must have type "
                    f"'{parameter.declared_type.describe()}', got "
                    f"'{argument_type.describe()}'",
                )

        return task_symbol.return_type

    def _type_check_list_literal(
        self,
        list_node: ASTNode,
        scope: Scope,
    ) -> SemanticType:
        element_type: SemanticType | None = None

        for item in list_node.value:
            item_type = self._type_check_expression(item, scope)

            if item_type.name == "unknown":
                continue

            if element_type is None:
                element_type = item_type
                continue

            if not self._same_type(element_type, item_type):
                self._error(
                    item,
                    f"List literal contains incompatible element types "
                    f"'{element_type.describe()}' and '{item_type.describe()}'",
                )

        return SemanticType("list", element_type)

    def _type_from_node(self, type_node: ASTNode) -> SemanticType:
        if type_node.value == "string":
            return STRING_TYPE
        if type_node.value == "int":
            return INT_TYPE
        if type_node.value == "bool":
            return BOOL_TYPE
        if type_node.value == "list":
            return LIST_TYPE

        self._error(type_node, f"Unsupported type '{type_node.value}'")

    def _merge_declared_and_value_type(
        self,
        declared_type: SemanticType,
        value_type: SemanticType,
    ) -> SemanticType:
        if declared_type.name == "list" and value_type.name == "list":
            return SemanticType("list", value_type.element_type)
        return declared_type

    def _is_assignable(
        self,
        target_type: SemanticType,
        value_type: SemanticType,
    ) -> bool:
        if target_type.name == "unknown" or value_type.name == "unknown":
            return True

        if target_type.name != value_type.name:
            return False

        if target_type.name != "list":
            return True

        if target_type.element_type is None or value_type.element_type is None:
            return True

        return self._is_assignable(target_type.element_type, value_type.element_type)

    def _are_comparable(
        self,
        left_type: SemanticType,
        right_type: SemanticType,
    ) -> bool:
        if left_type.name == "unknown" or right_type.name == "unknown":
            return True

        if left_type.name != right_type.name:
            return False

        if left_type.name != "list":
            return True

        if left_type.element_type is None or right_type.element_type is None:
            return True

        return self._are_comparable(left_type.element_type, right_type.element_type)

    def _same_type(self, left_type: SemanticType, right_type: SemanticType) -> bool:
        if left_type.name != right_type.name:
            return False

        if left_type.name != "list":
            return True

        if left_type.element_type is None or right_type.element_type is None:
            return left_type.element_type is right_type.element_type

        return self._same_type(left_type.element_type, right_type.element_type)

    def _error(self, node: ASTNode, message: str) -> NoReturn:
        if node.line is not None and node.column is not None:
            raise SemanticError(
                f"Semantic error at line {node.line}, column {node.column}: {message}"
            )

        raise SemanticError(f"Semantic error: {message}")
