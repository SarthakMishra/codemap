# CodeStruct VS Code Extension

This extension provides syntax highlighting for CodeStruct notation, a language-agnostic format for describing code structure.

## Features

- Syntax highlighting for CodeStruct notation
- Support for both standard and minified formats
- Custom file icon for CodeStruct files

## What is CodeStruct?

CodeStruct is a plain-text, human- and machine-readable, language-agnostic notation for describing the structure of software. It captures entities such as modules, classes, functions, parameters, variables, and more, in a concise, hierarchical, and extensible format. CodeStruct is designed for LLM context compression.

## Example

```
dir: project_root
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
```

## Minified Format Example

```
d:project_root;f:main.py;m:main;i:user[t:int,rf:m:user];fn:main|p:argv[t:LIST],r:None
```

## File Extensions

This extension supports the following file extensions:
- `.cstxt`
- `.codestruct`
- `.cs.txt`

## License

This extension is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 