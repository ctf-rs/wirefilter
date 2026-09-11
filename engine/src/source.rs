//! Parser-owned source metadata.

use crate::FilterAst;
use crate::ast::field_expr::lex_comparison_operator;
use crate::ast::index_expr::IndexExpr;
use crate::ast::logical_expr::LogicalOp;
use crate::ast::parse::FilterParser;
use crate::lex::{Lex, LexWith};

/// A half-open byte range in the original parser input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SourceRange {
    /// Inclusive byte offset where the range starts.
    pub start: usize,
    /// Exclusive byte offset where the range ends.
    pub end: usize,
}

impl SourceRange {
    /// Creates a source range from its byte offsets.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Returns the byte length of the range.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns whether the range is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// Source ranges belonging to one comparison expression.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ComparisonSource {
    /// The comparison's left-hand expression.
    pub lhs: SourceRange,
    /// The comparison operator, or an empty range for a bare boolean expression.
    pub operator: SourceRange,
    /// Right-hand literals or set members in parser order.
    pub values: Vec<SourceRange>,
}

/// The syntax category expected at a completion cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompletionExpectation {
    /// A field, function, unary operator, or opening parenthesis.
    Field,
    /// A comparison operator.
    Operator,
    /// A right-hand literal, list, or set member.
    Value,
    /// A logical operator joining complete expressions.
    Logical,
}

/// Parser-derived context for completing incomplete filter source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionContext {
    /// The syntax category expected at the cursor.
    pub expectation: CompletionExpectation,
    /// The preceding field or function identifier, when relevant.
    pub identifier: Option<String>,
    /// The source bytes that a completion should replace.
    pub replacement: SourceRange,
}

impl FilterParser<'_> {
    /// Analyzes incomplete filter source at a byte cursor.
    ///
    /// The cursor is clamped to the source and moved backwards to the nearest
    /// UTF-8 character boundary. When the cursor is inside a token, the
    /// replacement range covers the complete token where the parser can
    /// determine its end from the remaining source.
    #[must_use]
    pub fn completion_context(&self, source: &str, cursor: usize) -> CompletionContext {
        let cursor = floor_char_boundary(source, cursor.min(source.len()));
        let parser = self.with_source(source);
        let prefix = &source[..cursor];
        let input = prefix.trim();

        let _ = parser.lex_as::<FilterAst>(input);

        let mut context = parser.completion().unwrap_or(CompletionContext {
            expectation: CompletionExpectation::Field,
            identifier: None,
            replacement: SourceRange::new(cursor, cursor),
        });
        if prefix.trim_end().len() < prefix.len() {
            context.replacement = SourceRange::new(cursor, cursor);
        }

        refine_completion(self, source, cursor, context)
    }
}

fn refine_completion(
    parser: &FilterParser<'_>,
    source: &str,
    cursor: usize,
    mut context: CompletionContext,
) -> CompletionContext {
    if cursor >= source.len() {
        return context;
    }

    let full_parser = parser.with_source(source);
    let _ = full_parser.lex_as::<FilterAst>(source.trim_start());
    let comparisons = full_parser.comparison_sources();

    let parsed_range = match context.expectation {
        CompletionExpectation::Field => {
            shortest_containing(comparisons.iter().map(|comparison| comparison.lhs), cursor)
        }
        CompletionExpectation::Operator => shortest_containing(
            comparisons.iter().map(|comparison| comparison.operator),
            cursor,
        ),
        CompletionExpectation::Value => shortest_containing(
            comparisons
                .iter()
                .flat_map(|comparison| comparison.values.iter().copied()),
            cursor,
        ),
        CompletionExpectation::Logical => {
            let value = shortest_containing(
                comparisons
                    .iter()
                    .flat_map(|comparison| comparison.values.iter().copied()),
                cursor,
            );
            let operator = shortest_containing(
                comparisons.iter().map(|comparison| comparison.operator),
                cursor,
            );
            if let Some(range) = value {
                context.expectation = CompletionExpectation::Value;
                Some(range)
            } else if let Some(range) = operator {
                context.expectation = CompletionExpectation::Operator;
                Some(range)
            } else {
                None
            }
        }
    };

    if let Some(range) = parsed_range {
        context.replacement = range;
        return context;
    }

    let fragment = &source[context.replacement.start..];
    let range = match context.expectation {
        CompletionExpectation::Field => match IndexExpr::lex_with(fragment, &full_parser) {
            Ok((_, rest)) => full_parser.source_range(fragment, rest),
            Err(_) => None,
        },
        CompletionExpectation::Operator => lex_comparison_operator(fragment)
            .ok()
            .and_then(|rest| full_parser.source_range(fragment, rest)),
        CompletionExpectation::Logical => LogicalOp::lex(fragment)
            .ok()
            .and_then(|(_, rest)| full_parser.source_range(fragment, rest)),
        CompletionExpectation::Value => None,
    };

    if let Some(range) = range.filter(|range| {
        range.start <= cursor && cursor < range.end && context.replacement.start <= cursor
    }) {
        context.replacement = range;
    }
    context
}

fn shortest_containing(
    ranges: impl IntoIterator<Item = SourceRange>,
    cursor: usize,
) -> Option<SourceRange> {
    ranges
        .into_iter()
        .filter(|range| range.start <= cursor && cursor < range.end)
        .min_by_key(|range| range.len())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SourceOrigin {
    start: usize,
    len: usize,
}

impl SourceOrigin {
    pub(crate) fn new(source: &str) -> Self {
        Self {
            start: source.as_ptr() as usize,
            len: source.len(),
        }
    }

    pub(crate) fn range_of(self, slice: &str) -> Option<SourceRange> {
        let start = (slice.as_ptr() as usize).checked_sub(self.start)?;
        let end = start.checked_add(slice.len())?;
        (end <= self.len).then_some(SourceRange::new(start, end))
    }

    pub(crate) fn range_between(self, start: &str, rest: &str) -> Option<SourceRange> {
        let start = self.range_of(start)?.start;
        let end = self.range_of(rest)?.start;
        (start <= end).then_some(SourceRange::new(start, end))
    }
}

fn floor_char_boundary(source: &str, mut offset: usize) -> usize {
    while offset > 0 && !source.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LexErrorKind, SimpleFunctionArgKind, SimpleFunctionDefinition, SimpleFunctionImpl,
        SimpleFunctionParam, Type,
    };

    fn source_text(source: &str, range: SourceRange) -> &str {
        &source[range.start..range.end]
    }

    #[test]
    fn parse_errors_expose_structured_kinds_and_byte_spans() {
        let scheme = Scheme! {
            bytes: Bytes,
            int: Int,
        }
        .build();
        let source = "\n  bytes == \"é\" and\n  unknown";
        let error = scheme.parse(source).unwrap_err();
        let offset = source.find("unknown").unwrap();

        assert_eq!(error.kind(), &LexErrorKind::UnknownIdentifier);
        assert_eq!(
            error.span(),
            crate::ParseErrorSpan {
                line: 3,
                column: 3,
                length: 7,
                offset,
            }
        );

        let source = "  int ==";
        let error = scheme.parse(source).unwrap_err();
        assert_eq!(error.span().offset, source.len());
        assert_eq!(error.span().column, source.len() + 1);
        assert_eq!(error.span().length, 0);

        for whitespace in ["   ", "\t", "\u{2003}"] {
            let source = format!("int =={whitespace}");
            let error = scheme.parse(&source).unwrap_err();
            let span = error.span();
            assert_eq!(span.offset, source.len(), "{source:?}");
            assert_eq!(span.length, 0, "{source:?}");
            assert!(source.is_char_boundary(span.offset), "{source:?}");
            assert_eq!(error.to_string().matches('^').count(), 1, "{source:?}");
        }

        let source = "bytes == 'x'";
        let error = scheme.parse(source).unwrap_err();
        assert!(matches!(
            error.kind(),
            LexErrorKind::ParseInt { radix: 16, .. }
        ));
        assert_eq!(error.span().offset, source.find('\'').unwrap());
        assert_eq!(error.span().length, 2);
    }

    #[test]
    fn comparison_sources_cover_literals_sets_ranges_and_bare_booleans() {
        let scheme = Scheme! {
            bytes: Bytes,
            int: Int,
            ip: Ip,
            flag: Bool,
        }
        .build();
        let source = r###"  (bytes in {"plain" r#"raw é"# FF:FE FF:00} and ip in {dead::beef 2001:db8::/32 ::1..::2}) or (int in {1 2..4} and bytes == FF:00 and flag)"###;
        let ast = scheme.parse(source).unwrap();
        let comparisons = ast.comparison_sources();

        assert_eq!(comparisons.len(), 5);

        assert_eq!(source_text(source, comparisons[0].lhs), "bytes");
        assert_eq!(source_text(source, comparisons[0].operator), "in");
        assert_eq!(
            comparisons[0]
                .values
                .iter()
                .map(|&range| source_text(source, range))
                .collect::<Vec<_>>(),
            ["\"plain\"", "r#\"raw é\"#", "FF:FE", "FF:00"]
        );

        assert_eq!(source_text(source, comparisons[1].lhs), "ip");
        assert_eq!(source_text(source, comparisons[1].operator), "in");
        assert_eq!(
            comparisons[1]
                .values
                .iter()
                .map(|&range| source_text(source, range))
                .collect::<Vec<_>>(),
            ["dead::beef", "2001:db8::/32", "::1..::2"]
        );

        assert_eq!(source_text(source, comparisons[2].lhs), "int");
        assert_eq!(source_text(source, comparisons[2].operator), "in");
        assert_eq!(
            comparisons[2]
                .values
                .iter()
                .map(|&range| source_text(source, range))
                .collect::<Vec<_>>(),
            ["1", "2..4"]
        );

        assert_eq!(source_text(source, comparisons[3].lhs), "bytes");
        assert_eq!(source_text(source, comparisons[3].operator), "==");
        assert_eq!(source_text(source, comparisons[3].values[0]), "FF:00");

        assert_eq!(source_text(source, comparisons[4].lhs), "flag");
        assert!(comparisons[4].operator.is_empty());
        assert!(comparisons[4].values.is_empty());
    }

    #[test]
    fn nested_comparison_sources_follow_ast_traversal_order() {
        let mut builder = Scheme! {
            bytes: Bytes,
        };
        builder
            .add_function(
                "identity",
                SimpleFunctionDefinition {
                    params: vec![SimpleFunctionParam {
                        arg_kind: SimpleFunctionArgKind::Field,
                        val_type: Type::Bool,
                    }],
                    opt_params: vec![],
                    return_type: Type::Bool,
                    implementation: SimpleFunctionImpl::new(|args| args.next()?.ok()),
                },
            )
            .unwrap();
        let scheme = builder.build();
        let source = r#"identity(bytes == "x") and bytes == "y""#;
        let ast = scheme.parse(source).unwrap();
        let comparisons = ast.comparison_sources();

        assert_eq!(comparisons.len(), 3);
        assert_eq!(
            comparisons
                .iter()
                .map(|comparison| source_text(source, comparison.lhs))
                .collect::<Vec<_>>(),
            [r#"identity(bytes == "x")"#, "bytes", "bytes"]
        );
        assert!(comparisons[0].operator.is_empty());
        assert_eq!(source_text(source, comparisons[1].values[0]), r#""x""#);
        assert_eq!(source_text(source, comparisons[2].values[0]), r#""y""#);
    }

    #[test]
    fn completion_context_comes_from_parser_lexing() {
        let scheme = Scheme! {
            bytes: Bytes,
            int: Int,
            ip: Ip,
            flag: Bool,
        }
        .build();
        let parser = scheme.parser();

        let cases = [
            ("", CompletionExpectation::Field, None, ""),
            ("by", CompletionExpectation::Field, None, "by"),
            ("bytes ", CompletionExpectation::Operator, Some("bytes"), ""),
            (
                "bytes str",
                CompletionExpectation::Operator,
                Some("bytes"),
                "str",
            ),
            ("bytes == ", CompletionExpectation::Value, Some("bytes"), ""),
            (
                "bytes == r#\"é",
                CompletionExpectation::Value,
                Some("bytes"),
                "r#\"é",
            ),
            (
                "bytes == r#\"é\"# ",
                CompletionExpectation::Logical,
                Some("bytes"),
                "",
            ),
            (
                "int in {1 2..",
                CompletionExpectation::Value,
                Some("int"),
                "2..",
            ),
            (
                "int in {1 2..4} ",
                CompletionExpectation::Logical,
                Some("int"),
                "",
            ),
            (
                "bytes == FF:FE ",
                CompletionExpectation::Logical,
                Some("bytes"),
                "",
            ),
            (
                "bytes == FF:00 ",
                CompletionExpectation::Logical,
                Some("bytes"),
                "",
            ),
            (
                "ip == dead::beef ",
                CompletionExpectation::Logical,
                Some("ip"),
                "",
            ),
            (
                "int == 1 an",
                CompletionExpectation::Logical,
                Some("int"),
                "an",
            ),
            ("int == 1 && not ", CompletionExpectation::Field, None, ""),
            (
                "ip == dead::beef && by",
                CompletionExpectation::Field,
                None,
                "by",
            ),
            (
                "(bytes == \"x\" and (ip == dead::beef or ",
                CompletionExpectation::Field,
                None,
                "",
            ),
            ("flag ", CompletionExpectation::Logical, Some("flag"), ""),
        ];

        for (source, expectation, identifier, replacement) in cases {
            let context = parser.completion_context(source, source.len());
            assert_eq!(context.expectation, expectation, "{source:?}");
            assert_eq!(context.identifier.as_deref(), identifier, "{source:?}");
            assert_eq!(
                source_text(source, context.replacement),
                replacement,
                "{source:?}"
            );
        }
    }

    #[test]
    fn completion_cursor_is_clamped_to_utf8_boundaries() {
        let scheme = Scheme! {
            bytes: Bytes,
            int: Int,
            flag: Bool,
        }
        .build();
        let parser = scheme.parser();
        let source = "bytes == \"é\"";
        let inside_e_acute = source.find('é').unwrap() + 1;
        let context = parser.completion_context(source, inside_e_acute);

        assert_eq!(context.expectation, CompletionExpectation::Value);
        assert_eq!(context.identifier.as_deref(), Some("bytes"));
        assert_eq!(context.replacement.end, source.len());
        assert_eq!(source_text(source, context.replacement), "\"é\"");

        let context = parser.completion_context(source, 2);
        assert_eq!(context.expectation, CompletionExpectation::Field);
        assert_eq!(source_text(source, context.replacement), "bytes");

        let operator_start = source.find("==").unwrap();
        let context = parser.completion_context(source, operator_start + 1);
        assert_eq!(context.expectation, CompletionExpectation::Operator);
        assert_eq!(context.identifier.as_deref(), Some("bytes"));
        assert_eq!(source_text(source, context.replacement), "==");

        let source = "int == 123";
        let context = parser.completion_context(source, source.find('1').unwrap() + 1);
        assert_eq!(context.expectation, CompletionExpectation::Value);
        assert_eq!(context.identifier.as_deref(), Some("int"));
        assert_eq!(source_text(source, context.replacement), "123");

        let source = "bytes == \"x\" and flag";
        let logical_start = source.find("and").unwrap();
        let context = parser.completion_context(source, logical_start + 1);
        assert_eq!(context.expectation, CompletionExpectation::Logical);
        assert_eq!(context.identifier.as_deref(), Some("bytes"));
        assert_eq!(source_text(source, context.replacement), "and");

        let context = parser.completion_context(source, usize::MAX);
        assert_eq!(context.expectation, CompletionExpectation::Logical);
        assert_eq!(
            context.replacement,
            SourceRange::new(source.len(), source.len())
        );
    }

    #[test]
    fn completion_only_expands_tokens_containing_the_cursor() {
        let scheme = Scheme! {
            bytes: Bytes,
            int: Int,
            flag: Bool,
        }
        .build();
        let parser = scheme.parser();

        let source = "  bytes";
        let context = parser.completion_context(source, 1);
        assert_eq!(context.expectation, CompletionExpectation::Field);
        assert_eq!(context.replacement, SourceRange::new(1, 1));

        let source = "flag\t";
        let context = parser.completion_context(source, source.len());
        assert_eq!(context.expectation, CompletionExpectation::Logical);
        assert_eq!(
            context.replacement,
            SourceRange::new(source.len(), source.len())
        );

        let source = "int in {123";
        let cursor = source.find('1').unwrap() + 1;
        let context = parser.completion_context(source, cursor);
        assert_eq!(context.expectation, CompletionExpectation::Value);
        assert_eq!(source_text(source, context.replacement), "123");

        let source = "bytes == \"é\" unrelated";
        let cursor = source.find('é').unwrap() + 1;
        let context = parser.completion_context(source, cursor);
        assert_eq!(context.expectation, CompletionExpectation::Value);
        assert_eq!(source_text(source, context.replacement), "\"é\"");
    }

    #[test]
    fn mutable_walk_invalidates_filter_source_metadata() {
        use crate::{IndexExpr, VisitorMut};

        struct ClearIndexes(usize);

        impl<'a> VisitorMut<'a> for ClearIndexes {
            fn visit_index_expr(&mut self, node: &'a mut IndexExpr) {
                node.indexes.clear();
                self.0 += 1;
            }
        }

        let scheme = Scheme! {
            values: Array(Bytes),
        }
        .build();

        let mut ast = scheme.parse(r#"values[0] == "HIT""#).unwrap();
        assert_eq!(ast.comparison_sources().len(), 1);
        let mut visitor = ClearIndexes(0);
        ast.walk_mut(&mut visitor);
        assert_eq!(visitor.0, 1);
        assert!(ast.comparison_sources().is_empty());

        let mut ast = scheme.parse_value(r#"any(values[*] == "HIT")"#).unwrap();
        assert_eq!(ast.comparison_sources().len(), 1);
        let mut visitor = ClearIndexes(0);
        ast.walk_mut(&mut visitor);
        assert_eq!(visitor.0, 1);
        assert!(ast.comparison_sources().is_empty());
    }

    #[test]
    fn failed_function_parse_does_not_leak_comparison_sources() {
        let mut builder = Scheme! {
            values: Array(Bytes),
        };
        builder
            .add_function(
                "any",
                SimpleFunctionDefinition {
                    params: vec![SimpleFunctionParam {
                        arg_kind: SimpleFunctionArgKind::Literal,
                        val_type: Type::Int,
                    }],
                    opt_params: vec![],
                    return_type: Type::Bool,
                    implementation: SimpleFunctionImpl::new(|_| Some(true.into())),
                },
            )
            .unwrap();
        let scheme = builder.build();
        let source = r#"any(values[*] == "HIT")"#;
        let ast = scheme.parse_value(source).unwrap();
        let comparisons = ast.comparison_sources();

        assert_eq!(comparisons.len(), 1);
        assert_eq!(source_text(source, comparisons[0].lhs), "values[*]");
        assert_eq!(source_text(source, comparisons[0].operator), "==");
        assert_eq!(source_text(source, comparisons[0].values[0]), r#""HIT""#);
    }
}
