# CodeStruct Notation

## 1. Overview

**CodeStruct** is a plain-text, human- and machine-readable, language-agnostic notation for describing the structure of software. It captures entities such as modules, classes, functions, parameters, variables, and more, in a concise, hierarchical, and extensible format. CodeStruct is designed for LLM context compression.

---

## 2. Syntax Rules

### 2.1. Hierarchy and Indentation

- **Hierarchy** is represented by indentation (spaces or tabs; consistent usage is required).
- **Parent entities** contain **child entities** indented beneath them.

### 2.2. Entity Declaration

- Each entity is declared on a new line with the format:  
  ```
  :  []
  ```
- **Keywords** identify the entity type (see section 3).
- **Attributes** are optional, enclosed in square brackets, and comma-separated.

### 2.3. Attributes

- Attributes are key-value pairs: `key: value`
- Multiple attributes are separated by commas: `[type: INTEGER, default: 0]`

### 2.4. Documentation

- Use the `doc:` field after an entity to provide a one-line summary (first line of the docstring, truncated with `...` if needed).

### 2.5. Comments

- Lines starting with `#` are comments and ignored by parsers.

---

## 3. Supported Entity Keywords

| Keyword      | Description                                 | Example Usage                        |
|--------------|---------------------------------------------|--------------------------------------|
| `dir:`       | Directory (for filesystem hierarchy)         | `dir: src`                           |
| `file:`      | File                                        | `file: main.py`                      |
| `module:`    | Module or package                           | `module: user_management`            |
| `namespace:` | Namespace (for languages with namespaces)    | `namespace: std`                     |
| `class:`     | Class or type definition                    | `class: User`                        |
| `func:`      | Function or method                          | `func: login`                        |
| `lambda:`    | Lambda expression                           | `lambda: double`                     |
| `attr:`      | Attribute or class field                    | `attr: name [type: STRING]`          |
| `param:`     | Function/method parameter                   | `param: username [type: STRING]`     |
| `returns:`   | Function/method return type/value           | `returns: BOOLEAN`                   |
| `var:`       | Variable                                    | `var: counter [type: INTEGER]`       |
| `const:`     | Constant                                    | `const: PI [type: FLOAT]`            |
| `type_alias:`| Type alias                                  | `type_alias: UserID [type: INTEGER]` |
| `union:`     | Union type                                  | `union: Result`                      |
| `optional:`  | Optional value                              | `optional: nickname [type: STRING]`  |
| `import:`    | Import or dependency                        | `import: sys`                        |
| `doc:`       | Documentation summary                       | `doc: Handles user login...`         |

### 3.1. Import Classification

Imports can be classified as internal (defined within the same file) or external (from libraries or other modules):

| Field     | Purpose                                   | Example Usage                      |
|-----------|-------------------------------------------|-----------------------------------|
| `type:`   | Classification as `internal` or `external`| `type: internal`                  |
| `ref:`    | Reference to internal entity definition   | `ref: class: User`                |
| `source:` | Source for external imports              | `source: stdlib` or `source: pypi` |

Examples:
```
import: User
  type: internal
  ref: class: User

import: os
  type: external
  source: stdlib
```

---

## 4. Example

### 4.1. Directory and Code Structure

```codestruct
dir: project_root
  dir: src
    file: main.py
      module: main
        doc: Entry point for the application...
        import: user
          type: internal
          ref: module: user
        func: main
          doc: Runs the main application logic...
          param: argv [type: LIST]
          returns: None
    file: user.py
      module: user
        class: User
          doc: Represents a user...
          attr: name [type: STRING]
          attr: age [type: INTEGER]
          func: greet
            doc: Greets the user...
            returns: STRING
            import: os
              type: external
              source: stdlib
  dir: tests
    file: test_user.py
      func: test_greet
        doc: Tests the greet function...
        returns: None
  file: README.md
```

### 4.2. Import Classification Example

```
module: user_management
  class: User
    doc: Represents a user in the system...
    attr: name [type: STRING]
    attr: age [type: INTEGER]

  func: create_user
    doc: Creates a new user...
    param: name [type: STRING]
    param: age [type: INTEGER]
    returns: User
    import: User
      type: internal
      ref: class: User

  func: get_home_directory
    doc: Returns the user's home directory...
    returns: STRING
    import: os
      type: external
      source: stdlib
```

---

## 5. Reserved Keywords and Extensions

- Only listed keywords are reserved; users may extend with custom keywords as needed (e.g., `interface:`, `enum:`).
- Future extensions may add support for statements, decorators, or annotations.

---

## 6. Best Practices

- **Indentation:** Use 2 or 4 spaces for indentation; do not mix tabs and spaces.
- **Docstrings:** Always provide a `doc:` field for classes and functions for clarity.
- **Type Annotations:** Use `[type: ...]` for all parameters, attributes, and return values where possible.
- **Default Values:** Specify default values in attributes: `[default: ...]`.
- **Imports:** Use `import:` for dependencies at the file or module level.
  - Classify imports as `internal` or `external` using the `type:` field.
  - For internal imports, use `ref:` to link to the entity definition within the file.
  - For external imports, use `source:` to specify the origin (e.g., `stdlib`, `pypi`).

---

## 7. Parsing and Tooling

- CodeStruct is designed to be easily parsed by scripts or tools, enabling codebase analysis, documentation generation, and cross-language transformations.
- Parsers should ignore comments and handle missing optional fields gracefully.

---

## 8. Example with All Features

```
dir: my_project
  file: math_utils.py
    module: math_utils
      doc: Math utilities module...
      import: numpy
        type: external
        source: pypi
      func: add
        doc: Adds two numbers...
        param: a [type: INTEGER]
        param: b [type: INTEGER]
        returns: INTEGER
      func: multiply
        doc: Multiplies two numbers...
        param: a [type: INTEGER]
        param: b [type: INTEGER]
        returns: INTEGER
        import: add
          type: internal
          ref: func: add
      const: PI [type: FLOAT, default: 3.14159]
  file: README.md
```

---

## 9. Minification for LLM Context Compression

For scenarios requiring maximum context efficiency, CodeStruct can be minified while preserving its hierarchical structure and semantic information.

### 9.1. General Minification Principles

- Remove all blank lines.
- Replace indentation and line-breaks with delimiters: `;` to separate top-level entities, `|` to separate children, and `,` for attributes.
- Remove optional fields if not strictly needed.
- Use abbreviated forms for keywords (e.g., `cl:` for `class:`, `fn:` for `func:`, etc.).
- Truncate or omit docstrings where context is clear.
- Remove attribute names when the type provides sufficient context.
- Include a legend/mapping to help LLMs interpret the minified format correctly.

### 9.2. Keyword Shortening Reference

| Original    | Minified |
|-------------|----------|
| dir:        | d:       |
| file:       | f:       |
| module:     | m:       |
| class:      | cl:      |
| func:       | fn:      |
| attr:       | at:      |
| param:      | p:       |
| returns:    | r:       |
| var:        | v:       |
| const:      | c:       |
| import:     | i:       |
| doc:        | dc:      |
| type:       | t:       |
| default:    | d:       |
| source:     | s:       |
| ref:        | rf:      |

### 9.3. Attribute Compression

- Inline attributes with minimal separators:  
  `attr: name [type: STRING, default: "John"]` → `at:name[t:STR,d:"John"]`
- Use standard abbreviations for common types:
  - `INTEGER` → `INT`
  - `STRING` → `STR`
  - `BOOLEAN` → `BOOL`
  - `FLOAT` → `FLT`

### 9.4. Minification Example

Original CodeStruct:
```
dir: my_project
  file: math_utils.py
    module: math_utils
      doc: Math utilities module...
      import: numpy
        type: external
        source: pypi
      func: add
        doc: Adds two numbers...
        param: a [type: INTEGER]
        param: b [type: INTEGER]
        returns: INTEGER
      func: multiply
        doc: Multiplies two numbers...
        param: a [type: INTEGER]
        param: b [type: INTEGER]
        returns: INTEGER
        import: add
          type: internal
          ref: func: add
      const: PI [type: FLOAT, default: 3.14159]
  file: README.md
```

Minified Version:
```
d:my_project;f:math_utils.py;m:math_utils;i:numpy[t:ext,s:pypi];fn:add|p:a[t:INT],p:b[t:INT],r:INT;fn:multiply|p:a[t:INT],p:b[t:INT],r:INT,i:add[t:int,rf:fn:add];c:PI[t:FLT,d:3.14159];f:README.md
```

### 9.5. Structure Legend

When using minified CodeStruct, include a legend to assist LLMs:
```
Format: Entity;Entity|Child,Child[Attribute,Attribute]
Keyword map: d=dir,f=file,m=module,cl=class,fn=func,at=attr,p=param,r=returns,v=var,c=const,i=import,t=type,s=source,rf=ref
Type map: INT=INTEGER,STR=STRING,BOOL=BOOLEAN,FLT=FLOAT,ext=external,int=internal
Delimiters: ;=entity separator, |=child separator, ,=attribute separator, []=attribute container
```