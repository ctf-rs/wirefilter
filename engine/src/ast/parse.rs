use super::{FilterAst, FilterValueAst};
use crate::lex::{LexErrorKind, LexResult, LexWith, complete};
use crate::scheme::Scheme;
use crate::source::{
    ComparisonSource, CompletionContext, CompletionExpectation, SourceOrigin, SourceRange,
};
use std::cmp::{max, min};
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::{Arc, Mutex};

/// A filter parsing error associated with the original input.
#[derive(Debug, PartialEq)]
pub struct ParseError<'i> {
    /// The error that occurred when parsing the input
    pub(crate) kind: LexErrorKind,

    /// The input that caused the parse error
    pub(crate) input: &'i str,

    /// The line number on the input where the error occurred
    pub(crate) line_number: usize,

    /// The start of the bad input
    pub(crate) span_start: usize,

    /// The number of bytes that span the bad input on the displayed line
    pub(crate) span_len: usize,

    /// Absolute byte offset of the bad input
    pub(crate) absolute_span_start: usize,
}

/// A structured source span reported by the parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParseErrorSpan {
    /// 1-based line number.
    pub line: usize,
    /// 1-based byte column.
    pub column: usize,
    /// Byte length. Zero represents an insertion point, such as EOF.
    pub length: usize,
    /// Absolute byte offset in the original input.
    pub offset: usize,
}

impl Error for ParseError<'_> {}

impl<'i> ParseError<'i> {
    /// Create a new ParseError for the input, LexErrorKind and span in the
    /// input.
    pub fn new(mut input: &'i str, (kind, span): (LexErrorKind, &'i str)) -> Self {
        let source_range = SourceOrigin::new(input)
            .range_of(span)
            .unwrap_or_else(|| SourceRange::new(0, 0));
        let absolute_span_start =
            if span.is_empty() && input[source_range.start..].trim().is_empty() {
                input.len()
            } else {
                source_range.start
            };
        let mut span_start = absolute_span_start;

        let (line_number, line_start) = input[..span_start]
            .match_indices('\n')
            .map(|(pos, _)| pos + 1)
            .scan(0, |line_number, line_start| {
                *line_number += 1;
                Some((*line_number, line_start))
            })
            .last()
            .unwrap_or_default();

        input = &input[line_start..];

        span_start -= line_start;
        let mut span_len = span.len();

        if let Some(line_end) = input.find('\n') {
            input = &input[..line_end];
            span_len = min(span_len, line_end - span_start);
        }

        ParseError {
            kind,
            input,
            line_number,
            span_start,
            span_len,
            absolute_span_start,
        }
    }

    /// Returns the structured parser error kind.
    #[must_use]
    pub const fn kind(&self) -> &LexErrorKind {
        &self.kind
    }

    /// Returns the parser-owned location of the error in the original input.
    #[must_use]
    pub fn span(&self) -> ParseErrorSpan {
        ParseErrorSpan {
            line: self.line_number + 1,
            column: self.span_start + 1,
            length: self.span_len,
            offset: self.absolute_span_start,
        }
    }
}

impl Display for ParseError<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Filter parsing error ({}:{}):",
            self.line_number + 1,
            self.span_start + 1
        )?;

        writeln!(f, "{}", self.input)?;

        for _ in 0..self.span_start {
            write!(f, " ")?;
        }

        for _ in 0..max(1, self.span_len) {
            write!(f, "^")?;
        }

        writeln!(f, " {}", self.kind)?;

        Ok(())
    }
}

/// Parser settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParserSettings {
    /// Approximate size of the cache used by the DFA of a regex.
    /// Default: 10MB
    pub regex_dfa_size_limit: usize,
    /// Approximate size limit of the compiled regular expression.
    /// Default: 2MB
    pub regex_compiled_size_limit: usize,
    /// Maximum number of star metacharacters allowed in a wildcard.
    /// Default: unlimited
    pub wildcard_star_limit: usize,
    /// Maximum nesting depth allowed while parsing.
    /// Default: 128
    pub max_nesting_depth: u16,
}

impl Default for ParserSettings {
    #[inline]
    fn default() -> Self {
        Self {
            // Default value extracted from the regex crate.
            regex_compiled_size_limit: 10 * (1 << 20),
            // Default value extracted from the regex crate.
            regex_dfa_size_limit: 2 * (1 << 20),
            wildcard_star_limit: usize::MAX,
            max_nesting_depth: 128,
        }
    }
}

/// A structure used to drive parsing of an expression into a [`FilterAst`].
#[derive(Clone)]
pub struct FilterParser<'s> {
    pub(crate) scheme: &'s Scheme,
    pub(crate) settings: ParserSettings,
    current_nesting_depth: u16,
    source: Option<ParserSource>,
}

#[derive(Clone, Debug)]
struct ParserSource {
    origin: SourceOrigin,
    state: Arc<Mutex<ParserSourceState>>,
}

#[derive(Debug, Default)]
struct ParserSourceState {
    comparisons: Vec<ComparisonSource>,
    completion: Option<CompletionContext>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParserSourceCheckpoint {
    comparisons_len: usize,
    completion: Option<CompletionContext>,
}

impl Debug for FilterParser<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterParser")
            .field("scheme", &self.scheme)
            .field("settings", &self.settings)
            .field("current_nesting_depth", &self.current_nesting_depth)
            .finish()
    }
}

impl PartialEq for FilterParser<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.scheme == other.scheme
            && self.settings == other.settings
            && self.current_nesting_depth == other.current_nesting_depth
    }
}

impl Eq for FilterParser<'_> {}

impl<'s> FilterParser<'s> {
    /// Creates a new parser with default settings.
    #[inline]
    pub fn new(scheme: &'s Scheme) -> Self {
        Self {
            scheme,
            settings: ParserSettings::default(),
            current_nesting_depth: 0,
            source: None,
        }
    }

    /// Creates a new parser with the specified settings.
    #[inline]
    pub fn with_settings(scheme: &'s Scheme, settings: ParserSettings) -> Self {
        Self {
            scheme,
            settings,
            current_nesting_depth: 0,
            source: None,
        }
    }

    /// Returns the [`Scheme`](struct@Scheme) for which this parser has been constructor for.
    #[inline]
    pub fn scheme(&self) -> &'s Scheme {
        self.scheme
    }

    #[inline]
    pub(crate) fn lex_as<'i, L: for<'p> LexWith<'i, &'p FilterParser<'s>>>(
        &self,
        input: &'i str,
    ) -> LexResult<'i, L> {
        L::lex_with(input, self)
    }

    #[inline]
    pub(crate) fn with_increased_nesting<'i>(
        &self,
        span: &'i str,
    ) -> Result<Self, (LexErrorKind, &'i str)> {
        if self.current_nesting_depth >= self.settings.max_nesting_depth {
            Err((
                LexErrorKind::NestingLimitExceeded {
                    limit: self.settings.max_nesting_depth,
                },
                span,
            ))
        } else {
            let mut nested = self.clone();
            nested.current_nesting_depth += 1;
            Ok(nested)
        }
    }

    pub(crate) fn with_source(&self, source: &str) -> Self {
        Self {
            scheme: self.scheme,
            settings: self.settings.clone(),
            current_nesting_depth: 0,
            source: Some(ParserSource {
                origin: SourceOrigin::new(source),
                state: Arc::default(),
            }),
        }
    }

    pub(crate) fn source_range(&self, start: &str, rest: &str) -> Option<SourceRange> {
        self.source.as_ref()?.origin.range_between(start, rest)
    }

    pub(crate) fn begin_comparison_source(
        &self,
        lhs: Option<SourceRange>,
        operator: Option<SourceRange>,
    ) {
        let (Some(parser_source), Some(lhs), Some(operator)) = (&self.source, lhs, operator) else {
            return;
        };
        parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .comparisons
            .push(ComparisonSource {
                lhs,
                operator,
                values: Vec::new(),
            });
    }

    pub(crate) fn update_comparison_operator_source(&self, operator: Option<SourceRange>) {
        let (Some(parser_source), Some(operator)) = (&self.source, operator) else {
            return;
        };
        if let Some(comparison) = parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .comparisons
            .last_mut()
        {
            comparison.operator = operator;
        }
    }

    pub(crate) fn record_comparison_value_source(&self, value: SourceRange) {
        let Some(parser_source) = &self.source else {
            return;
        };
        if let Some(comparison) = parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .comparisons
            .last_mut()
        {
            comparison.values.push(value);
        }
    }

    pub(crate) fn finish_comparison_source(&self, source: ComparisonSource) {
        let Some(parser_source) = &self.source else {
            return;
        };
        let mut state = parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(comparison) = state.comparisons.last_mut()
            && comparison.lhs == source.lhs
        {
            *comparison = source;
        } else {
            state.comparisons.push(source);
        }
    }

    pub(crate) fn comparison_sources(&self) -> Vec<ComparisonSource> {
        let Some(parser_source) = &self.source else {
            return Vec::new();
        };
        let mut comparisons = parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .comparisons
            .clone();
        comparisons.sort_by_key(|source| (source.lhs.start, std::cmp::Reverse(source.lhs.end)));
        comparisons
    }

    pub(crate) fn source_checkpoint(&self) -> Option<ParserSourceCheckpoint> {
        self.source.as_ref().map(|parser_source| {
            let state = parser_source
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ParserSourceCheckpoint {
                comparisons_len: state.comparisons.len(),
                completion: state.completion.clone(),
            }
        })
    }

    pub(crate) fn restore_source(&self, checkpoint: Option<ParserSourceCheckpoint>) {
        let (Some(parser_source), Some(checkpoint)) = (&self.source, checkpoint) else {
            return;
        };
        let mut state = parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.comparisons.truncate(checkpoint.comparisons_len);
        state.completion = checkpoint.completion;
    }

    pub(crate) fn record_completion(
        &self,
        expectation: CompletionExpectation,
        identifier: Option<&str>,
        replacement: &str,
    ) {
        let Some(parser_source) = &self.source else {
            return;
        };
        let Some(replacement) = parser_source.origin.range_of(replacement) else {
            return;
        };
        let mut state = parser_source
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let should_replace = state.completion.as_ref().is_none_or(|current| {
            current.replacement.start < replacement.start
                || (current.replacement.start == replacement.start
                    && (current.identifier.is_none() || identifier.is_some()))
        });
        if should_replace {
            state.completion = Some(CompletionContext {
                expectation,
                identifier: identifier.map(str::to_owned),
                replacement,
            });
        }
    }

    pub(crate) fn completion(&self) -> Option<CompletionContext> {
        self.source.as_ref().and_then(|parser_source| {
            parser_source
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .completion
                .clone()
        })
    }

    /// Parses a filter expression into an AST form.
    pub fn parse<'i>(&self, input: &'i str) -> Result<FilterAst, ParseError<'i>> {
        let parser = self.with_source(input);
        complete(parser.lex_as(input.trim())).map_err(|err| ParseError::new(input, err))
    }

    /// Parses a value expression into an AST form.
    pub fn parse_value<'i>(&self, input: &'i str) -> Result<FilterValueAst, ParseError<'i>> {
        let parser = self.with_source(input);
        complete(parser.lex_as(input.trim())).map_err(|err| ParseError::new(input, err))
    }

    /// Retrieve parser settings.
    #[inline]
    pub fn settings(&self) -> &ParserSettings {
        &self.settings
    }

    /// Set the approximate size limit of the compiled regular expression.
    #[inline]
    pub fn regex_set_compiled_size_limit(&mut self, regex_compiled_size_limit: usize) {
        self.settings.regex_compiled_size_limit = regex_compiled_size_limit;
    }

    /// Get the approximate size limit of the compiled regular expression.
    #[inline]
    pub fn regex_get_compiled_size_limit(&self) -> usize {
        self.settings.regex_compiled_size_limit
    }

    /// Set the approximate size of the cache used by the DFA of a regex.
    #[inline]
    pub fn regex_set_dfa_size_limit(&mut self, regex_dfa_size_limit: usize) {
        self.settings.regex_dfa_size_limit = regex_dfa_size_limit;
    }

    /// Get the approximate size of the cache used by the DFA of a regex.
    #[inline]
    pub fn regex_get_dfa_size_limit(&self) -> usize {
        self.settings.regex_dfa_size_limit
    }

    /// Set the maximum number of star metacharacters allowed in a wildcard.
    #[inline]
    pub fn wildcard_set_star_limit(&mut self, wildcard_star_limit: usize) {
        self.settings.wildcard_star_limit = wildcard_star_limit;
    }

    /// Get the maximum number of star metacharacters allowed in a wildcard.
    #[inline]
    pub fn wildcard_get_star_limit(&self) -> usize {
        self.settings.wildcard_star_limit
    }

    /// Set the maximum nesting depth allowed while parsing.
    #[inline]
    pub fn set_max_nesting_depth(&mut self, max_nesting_depth: u16) {
        self.settings.max_nesting_depth = max_nesting_depth;
    }

    /// Get the maximum nesting depth allowed while parsing.
    #[inline]
    pub fn max_nesting_depth(&self) -> u16 {
        self.settings.max_nesting_depth
    }
}
