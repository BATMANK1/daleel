# Daleel

Answers questions about Royal Commission for Jubail and Yanbu college
regulations in Arabic, with the exact clause cited.

> **Status: in development.** This README grows with the repository.
> No result is published here until it has been measured.

## The problem

Students need answers from 137 pages of Arabic regulation documents:
*Can I transfer majors? What GPA do I need? When is the drop/add deadline?*
Today the options are reading all of it, asking a classmate who may be wrong,
or waiting for the registrar.

An unsourced answer about your own academic standing is unusable — you cannot
act on it and the registrar cannot confirm it. So **every answer this system
gives carries a document, page and clause reference.** Citation is a hard
functional requirement, not a feature.

## Why this is not a "chat with your PDF" project

Measuring the corpus before building anything turned up three distinct
extraction failure modes across the five documents:

1. **Presentation-form substitution.** 62% of the Arabic in the main guide
   extracts as Unicode Arabic Presentation Forms rather than base letters,
   alongside 11,332 bidirectional control characters. A student typing a word
   never matches an index holding its contextual glyph forms.
2. **Silently wrong text.** Two documents extract as *valid Arabic codepoints
   that are not real words* — letters substituted and dropped, with nothing
   structural to flag it.
3. **Corrupted codepoints.** The academic calendar's text layer returns raw
   control characters where Arabic letters should be.

Every naive pipeline built on this corpus fails, and fails *silently* —
retrieving nothing relevant and generating a confident answer anyway. So
extraction here is a component with its own tests and its own quality gate,
not a preprocessing line.

## Results

Not yet measured. Tables land here as the work completes.

## Licensing

The source documents are internal college publications and are not
redistributed here. See [`data/README.md`](data/README.md) for how to obtain
them.
