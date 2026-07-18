# SciFact attribution and data notice

The Late Chunking Deep Dive uses the SciFact test retrieval benchmark introduced
by David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman
Cohan, and Hannaneh Hajishirzi in [*Fact or Fiction: Verifying Scientific Claims*](https://arxiv.org/abs/2004.14974).
SciFact was created by the Allen Institute for AI. The frozen experiment downloads
the SciFact conversion distributed with the [BEIR benchmark](https://github.com/beir-cellar/beir).

The committed `receipts/scifact-test-qrels.tsv` file is a byte-identical copy of
the frozen archive's `qrels/test.tsv`; it is not modified. Other committed receipts
contain opaque SciFact query or document identifiers, relevance labels, rankings,
and derived metrics. They do not redistribute the claim text or corpus abstracts.

Under the [SciFact data license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md):

- SciFact claims and evidence annotations are licensed under the
  [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
- The collection of scientific-paper abstracts is licensed under the
  [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).

The embedding scores, rankings, metrics, receipts, and figures in this Deep Dive
are new derived results produced by the bundled reproduction script. Their presence
does not imply endorsement by the SciFact authors, the Allen Institute for AI, or
the BEIR authors.
