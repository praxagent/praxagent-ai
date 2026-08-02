---
title: "The Mathematics of Dense Vector Search"
slug: "vector-search-math"
date: 2026-02-17
lastmod: 2026-07-27
author: Timothy Jones
summary: "A comprehensive, first-principles climb through dense vector search: exact scoring, high-dimensional geometry, KD-trees and R-trees, ANN, HNSW, DiskANN and Vamana, IVF, quantization, non-determinism, evaluation, and production selection."
weight: 30
draft: true
pro_reviewed: true
og_image: "og-card.png"
og_image_alt: "A proximity graph crosses a search landscape. A dashed one-candidate route stops in a nearby basin, while a solid wider-candidate route reaches the deeper target basin."
ai_disclosure: |
  **Artificial-intelligence (AI) use disclosure.** Generative-AI tools helped expand, audit,
  illustrate, review, and edit this Deep Dive. The human author selected the
  teaching goals, shaped the full technical scope, and is
  responsible for the final text, code, citations, and claims. This is an
  independent teaching guide, not a peer-reviewed paper or vendor benchmark.
---

Consider these two sentences:


> The weather was amazing today and made me so happy!

> My day was made all the better by the beautiful weather!

Our minds immediately see the similarity between them, such that if someone were to
speak them to us, one after the other, we would understand that they were repeating
for emphasis to express the magnitude of the emotional valence. We recognize their
semantic similiarity, that their meaning is very similar.

Now consider these two sentences:

> The weather was amazing today and made me so happy!

> My car broke down and I got a flat tire on the way to work!

These are clearly two sentences whose meaning is not similar. Hearing these two 
sentences together would construct a narrative where an individual was having a
great day and then it took a sour turn.

We recognize these similarities and dissimilarities from decades of practice, 
education, and lived experience. How can this skill be taught to thinking machines?

The answer involves transforming sentences into mathematical vectors whose direction
in a high dimensional space represents an abstraction of their meaning.

**Dense vector search** ranks stored items represented by fixed-length vectors
whose coordinate values are generally stored explicitly. Unlike a sparse
vector, a dense vector does not represent an item as only a relatively small
set of nonzero coordinates. An {{< refterm "embedding" "embedding" >}} model
creates those vectors. Their geometry preserves relationships that the search
system can use.

Modern systems often use vector search with large language models (LLMs),
semantic search, and retrieval-augmented generation (RAG). Image-text models
such as Contrastive Language-Image Pre-training (CLIP) also use vectors. A
vector database can retrieve domain-specific items without an exact substring
match. However, the system does not read meaning directly. It applies a
mathematical comparison rule to numbers from a specified model.

For example, an image embedding model is useful only if its geometry preserves
relationships that matter for the task. Two dog images should usually be nearer
each other than a dog image and a car image. A text embedding model has the
same teaching goal for related sentences:

> The weather was amazing today and made me so happy!

> My day was made all the better by the beautiful weather!

{{< reference-figure
  src="vector_viz_close.png"
  label="Two related sentence vectors"
  alt="A two-dimensional teaching plot shows a weather query vector and a related candidate vector. The vectors point in nearly the same direction. Their cosine similarity is 0.997. Their angle is 4.5 degrees, and their Euclidean distance is 0.078."
  caption="The query and related candidate use hand-authored teaching coordinates. Their directions nearly agree. These are not outputs from a named embedding model and do not measure universal semantic similarity. [reproduce.py](reproduce.py) generated the figure. The [figure receipt](fig-vector-search-teaching.receipt.json) records its hashes."
>}}

Now compare the first weather sentence with a sentence about a car problem:

> The weather was amazing today and made me so happy!

> My car broke down and I got a flat tire on the way to work!

{{< reference-figure
  src="vector_viz_far.png"
  label="Two unrelated sentence vectors"
  alt="A two-dimensional teaching plot shows a weather query vector in the upper right. An unrelated candidate vector is in the lower left. Their cosine similarity is negative 0.960. Their angle is 163.7 degrees, and their Euclidean distance is 1.980."
  caption="The figure compares the same query with an unrelated candidate. Direction and distance now disagree strongly with the query. A real result depends on the model, input text, pooling rule, normalization, and comparison rule. The pooling rule combines token-level representations into one item vector. [reproduce.py](reproduce.py) generated the figure. The [figure receipt](fig-vector-search-teaching.receipt.json) records its hashes."
>}}

In our dummy example, let the query vector be
\(q=(0.80,0.60)\). The symbol \(q\) means *query*. Assign these hand-authored
teaching vectors to the three candidate sentences:

| Candidate | Teaching vector |
|---|---:|
| related weather sentence | \((0.75,0.66)\) |
| sentence pointing straight up | \((0.00,1.00)\) |
| unrelated car-breakdown sentence | \((-0.60,-0.80)\) |

Two coordinates let us draw the geometry without claiming that either visible
axis has a human-readable meaning. The values are illustrative, not outputs
from a named model. The calculations below use these values to produce exact
distances and rankings.

These examples give only intuition. We must define *near* and compute exact
answers. We must also learn why classical shortcuts fail. Then we can state
what approximate nearest-neighbor (ANN) algorithms can miss.

{{< panel "info" >}}
**Learning path**

1. Build one exact answer from coordinates you can check by hand.
2. Learn how KD-trees and R-trees prune low-dimensional space.
3. Watch those pruning certificates weaken as dimension grows.
4. Define the ANN accuracy contract with exact-neighbor recall.
5. Build HNSW slowly: layers, promotion, graph construction, greedy descent,
   the bottom-layer candidate queues, complexity, and failure modes.
6. Follow the same design pressures into DiskANN/Vamana, IVF, PQ, SQ, BQ,
   Fast Scan, and Scalable Nearest Neighbors (ScaNN).
7. Finish with non-determinism, filtered and hybrid retrieval, benchmark
   design, implementation choices, and a production selection matrix.
{{< /panel >}}

## Summary: the 60-second version

- **The problem:** High dimension can weaken the geometric pruning rules used
  by classical spatial trees. On some distributions, near and far distances
  also lose relative contrast. Neither effect proves that every embedding
  collection is unsearchable; both make the actual distribution important.
- **The graph route:** **Hierarchical navigable small world (HNSW)** search uses
  a layered proximity graph. Sparse upper layers make long moves; the complete
  bottom layer searches a wider candidate set. The HNSW paper reports
  logarithmic empirical scaling, not a worst-case guarantee.
- **The storage challenge:** HNSW stores graph links in addition to vectors.
  Link width, \(M\), bottom-layer capacity, allocator layout, vector format, and
  implementation decide whether the graph or the vector payload dominates
  random-access memory (RAM).
- **The storage options**:
    - **Quantization:** Product quantization (PQ), scalar quantization (SQ), and
      binary quantization (BQ) trade stored precision for cheaper candidate
      scoring. Under the worked setup below, PQ changes a 384-coordinate
      float32 payload from 1,536 bytes to 48 bytes; total index size is larger.
    - **Disk-aware graphs:** DiskANN and its Vamana graph use compressed
      in-memory guidance, caching, and solid-state-drive (SSD) access. They do
      not require the full graph and all full-precision vectors to fit in RAM.
- **The contract:** Each shortcut needs evaluation against exact top-\(k\)
  neighbors and application relevance. Recall, tail latency,
  throughput, memory, build cost, filters, updates, and cache state belong in
  the same decision.
- **Sparse retrieval:** Lexical systems such as Okapi BM25 usually use
  inverted indexes. Learned sparse methods such as the Sparse
  Lexical and Expansion Model (SPLADE) also use them.
  Their sparse access pattern has different costs and failure modes from the
  dense ANN pipelines.

> **Note:** Code blocks in this document are minimal for conceptual clarity; some explicitly omit imports, memory management, and structural boilerplate.

---

## Scope: dense vectors, not sparse retrieval
This Deep Dive focuses on the mathematics and systems design of **dense vector
search**.

A dense embedding stores a value for every coordinate in its fixed-width
representation, so a direct comparison normally touches every coordinate. A
**sparse vector** can have a much larger mathematical dimension while storing
only the coordinates whose values are nonzero.

Traditional term-frequency/inverse-document-frequency (TF-IDF) retrieval,
BM25, and learned sparse representations such as
[SPLADE](https://arxiv.org/abs/2107.05720) use sparse term or expansion
features. Think of each feature as a coordinate that is usually zero. A
sentence such as "rain today" may set only the coordinates for `rain` and
`today` (and any expansion terms the model adds). Every other vocabulary
coordinate stays zero and need not be stored.

An **inverted index** flips the usual item-to-features table. Instead of
asking "which terms appear in document 3?", it answers "which documents
contain the term `rain`?". The list of item identifiers for one feature is
called a **posting list**, or simply the **postings** for that feature.

A tiny collection makes the shortcut concrete:

| Item | Stored nonzero features |
|---|---|
| Doc A | `rain`, `today` |
| Doc B | `car`, `battery` |
| Doc C | `rain`, `forecast` |

The inverted index stores the reverse map:

| Feature | Postings |
|---|---|
| `rain` | A, C |
| `today` | A |
| `car` | B |
| `battery` | B |
| `forecast` | C |

For the query "rain today", search opens only the postings for `rain` and
`today`. It scores Doc A and Doc C. It never opens Doc B, and it never walks
the full vocabulary coordinate-by-coordinate for every document. Dense
embedding search does not get this free skip: a direct dense comparison
touches every coordinate of every candidate it evaluates.

That sparse access pattern does not make sparse search immune to scale, skew,
high-frequency terms, or ranking error. It is a different problem from dense
vector search, with different data structures and failure modes. This Deep Dive
stays with dense vectors: exact scoring, high-dimensional geometry, classical
spatial trees, and the approximate methods used when a direct comparison must
touch every coordinate.

## The classical approach: partition space

The first acceleration strategy is exact, not approximate: divide space into
regions, then prove that some regions cannot contain a better answer.

### KD-trees: split one coordinate at a time

A **k-dimensional tree (KD-tree)** is a binary tree for points that live in
\(k\)-dimensional space. The \(k\) is that coordinate count: two for a plane,
three for ordinary space, \(384\) for a typical embedding, and so on. Each internal
node chooses one of those \(k\) coordinates and one split value. An
**axis-aligned hyperplane** is the higher-dimensional version of a vertical or
horizontal cut. The cut fixes one coordinate and does not restrict the other
coordinates.

{{< reference-figure
  src="kd-tree-axis-aligned-split.svg"
  label="Axis-aligned KD-tree splits"
  alt="Two panels of the same eight toy points in the plane. The left panel shows one vertical cut at x equals 50, separating points with x less than 50 from points with x at least 50 while leaving y free. The right panel keeps that vertical cut and adds a horizontal cut inside each side: y equals 30 on the left and y equals 80 on the right. Points A and B lie below y equals 30; C and D lie above it. Points E and F lie below y equals 80; G and H lie above it."
  caption="In two dimensions, an axis-aligned cut is just a vertical or horizontal line. The root chooses one coordinate and one split value. The next level cycles to another coordinate and can use a different local median in each branch. These toy points and thresholds match the tree questions in the figure below. In higher dimensions the same idea becomes a hyperplane that still fixes only one coordinate."
>}}

The diagram above is the two-coordinate case. Building the tree is just
repeating that cut, one coordinate at a time.

Start with every point in one bucket. Sort that bucket by the current
coordinate and pick a **median** point near the middle of that sorted list.
Store the median at the current node. Points with a smaller value on that
coordinate go to the left child. Points with a larger value go to the right
child. If two points tie on the split coordinate, pick a fixed rule and stick
to it; the teaching code below may place a tied point on either side.

Then move one level deeper and change which coordinate you sort on:

1. **Depth 0 (root):** sort and split on coordinate \(0\) (\(x\) in the
   diagram). That is the vertical cut.
2. **Depth 1:** inside each child, sort and split on coordinate \(1\)
   (\(y\)). That is why the left branch can use \(y=30\) while the right
   branch uses \(y=80\).
3. **Depth 2 and later:** if the points have a third coordinate, split on
   coordinate \(2\) next. After the last coordinate, cycle back to
   coordinate \(0\), then \(1\), and so on.

So a three-coordinate dataset uses the repeating axis order
\(0,1,2,0,1,2,\ldots\). A two-coordinate dataset, like the diagram, uses
\(0,1,0,1,\ldots\). The code rule is simply `axis = depth % k`.

The following code builds the tree. It does not yet implement nearest-neighbor
search.

**Build cost.** A full sort at every level costs \(O(n\log n)\) work on that
level. With \(O(\log n)\) levels, the total becomes \(O(n\log^2 n)\). The
builders below avoid that. They need only the median on the current axis, so
they use **selection** (partition until the median index is correct). A
subtree of \(m\) points costs expected \(O(m)\) time, and the whole build is
expected \(O(n\log n)\). They also rearrange one shared buffer in place, so
working storage for the points stays \(O(n)\).

A slicing-based recursive builder may copy \(O(n\log n)\) point references or
values **cumulatively** over the complete build, because each tree level
processes a total of \(O(n)\) elements. That is an allocation-work bound, not
normally the peak live-memory bound. In a conventional depth-first
implementation, temporary slices along one active recursion path and its
pending siblings usually total \(O(n)\) live elements, plus an \(O(\log n)\)
call stack for a balanced tree. Peak memory can become larger if an
implementation retains intermediate arrays or eagerly materializes additional
copies. The in-place range-based builder below avoids that copying and
allocation churn while retaining \(O(n)\) point storage.

**Duplicate coordinates.** A two-way partition that tests only
`value < pivot` never places equals into the left block. That matters when
every point in a subtree shares the same axis value (grids, quantized sensors,
rounded timestamps).

Walk through that bad case with a tiny teaching set. All four points share the
same \(x\) value; only \(y\) differs:

```
dupes = [(5, 1), (5, 2), (5, 3), (5, 4)]
```

Here \(m=4\). Split on axis \(0\) (the \(x\) coordinate) and aim for the median
index `mid = 2`. Every \(x\) is \(5\), so a two-way test `x < pivot` never
succeeds.

1. Pass 1 scans all 4 points, finds nothing smaller, parks the pivot at index
   `0`, and shrinks the range to the other 3 points.
2. Pass 2 scans those 3 points, shrinks to 2.
3. Pass 3 scans 2 points, shrinks to 1.
4. Pass 4 has a single point left.

Work adds up as \(4+3+2+1=10\). In general, with \(m\) equal values you pay

\[
m + (m-1) + (m-2) + \cdots + 1 = \frac{m(m+1)}{2},
\]

which is \(O(m^2)\). A random pivot does not help: every \(x\) is \(5\), so
every pivot gives the same useless split.

All three builders therefore use a **three-way partition**: smaller, equal, and
larger. When the target index lands in the equal block, selection stops after
one \(O(m)\) pass even if every value is identical.

### How the Python builder fits together

Read the four functions as one pipeline. The public entry point prepares a
buffer. The recursive builder chooses a median index. Selection moves that
median into place. The three-way partition is the rearrange step selection
calls.

{{< mermaid >}}
flowchart TD
    A["build_kdtree(points)<br/>entry point"] --> B["Copy rows into buf<br/>record k = dimension"]
    B --> C["_build_kdtree(buf, start, end, depth, k)<br/>one subtree range"]
    C --> D{"Range empty?"}
    D -->|yes| E["Return None"]
    D -->|no| F["axis = depth % k<br/>mid = middle index"]
    F --> G["_select(... mid ...)<br/>put a median value at mid"]
    G --> H["_partition3(...)<br/>smaller | equal | larger"]
    H --> G
    G --> I["Make Node at buf[mid]"]
    I --> J["_build_kdtree left<br/>[start, mid)"]
    I --> K["_build_kdtree right<br/>(mid, end)"]
    J --> C
    K --> C
{{< /mermaid >}}

<p class="figure-note">Figure: call flow for the Python builder. Thick idea:
<code>_build_kdtree</code> owns the tree shape; <code>_select</code> owns
finding the median index; <code>_partition3</code> owns one rearrange of the
shared buffer. Left and right recurse on index ranges of that same buffer.</p>

{{< reference-figure
  src="kd-tree-partition3.svg"
  label="Three-way partition blocks"
  alt="Two rows of seven axis values. Before partition the values are unsorted and one cell marked 5 is the pivot. After partition the array reads 2, 1, 4, then 5, 5, then 9, 7. Brackets label the smaller block ending at lt, the equal block from lt to gt, and the larger block starting at gt. A target marker sits on an equal-block cell."
  caption="What `_partition3` does in one picture. It does not sort the whole range. It only groups values into smaller, equal, and larger on the current axis. `_select` picks a random pivot, calls this rearrange, then either stops (target inside the equal block) or repeats on the side that still contains the target index."
>}}

In short:

1. **`build_kdtree`**: convert inputs once; remember \(k\); start recursion on
   the full range.
2. **`_build_kdtree`**: for the current index range, choose the split axis and
   the median index `mid`, call `_select`, store `points[mid]` in a node, then
   recurse on the left and right index ranges.
3. **`_select`**: keep partitioning until `mid` sits inside an equal block, so
   `points[mid]` holds the median axis value.
4. **`_partition3`**: one linear scan that creates the three blocks and returns
   `(lt, gt)`.

Let `points` be a small teaching set:

```
points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
```

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import random

class Node:
    def __init__(self, point, left=None, right=None, axis=0):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

def build_kdtree(points):
    # Entry point: convert once to a list of tuples.
    # A NumPy ndarray works here because we iterate rows into tuples.
    # Do not run the helpers on a raw ndarray: `if not arr` is ambiguous, and
    # ndarray.sort does not accept key=. For large arrays, prefer a separate
    # NumPy path (for example np.argpartition) rather than this conversion.
    buf = [tuple(p) for p in points]
    if not buf:
        return None
    k = len(buf[0])
    if k == 0:
        raise ValueError("points must have at least one coordinate")
    return _build_kdtree(buf, 0, len(buf), 0, k)

def _partition3(points, left, right, axis, pivot_index):
    """Rearrange points[left:right+1] into three blocks on the chosen axis:
    smaller, then equal, then larger.

    Return (lt, gt). The equal block is points[lt:gt].
    Any index inside that block holds a correct median value.
    """
    pivot_value = points[pivot_index][axis]
    lt = left        # end of the "smaller" block
    i = left         # next unexamined slot
    gt = right + 1   # start of the "larger" block

    while i < gt:
        v = points[i][axis]
        if v < pivot_value:
            # Swap into the growing smaller block, then advance.
            points[lt], points[i] = points[i], points[lt]
            lt += 1
            i += 1
        elif v > pivot_value:
            # Swap with the growing larger block on the right.
            # Do not advance i: the point swapped in is still unexamined.
            gt -= 1
            points[gt], points[i] = points[i], points[gt]
        else:
            i += 1   # equal: leave it in the middle block

    return lt, gt

def _select(points, left, right, target, axis):
    """Ensure points[target] holds the target-th smallest value on this axis.

    Expected O(m) for m = right - left + 1, including on duplicate values.
    """
    while left < right:
        pivot_index = random.randint(left, right)
        lt, gt = _partition3(points, left, right, axis, pivot_index)
        if target < lt:
            right = lt - 1   # target lies among the smaller values
        elif target >= gt:
            left = gt        # target lies among the larger values
        else:
            return           # target landed inside the equal block

def _build_kdtree(points, start, end, depth, k):
    # In-place index ranges on one buffer: no per-level copy or slice.
    n = end - start
    if n <= 0:
        return None

    axis = depth % k          # which coordinate this level splits on
    mid = start + n // 2      # median index inside [start, end)
    _select(points, start, end - 1, mid, axis)

    return Node(
        point=points[mid],
        left=_build_kdtree(points, start, mid, depth + 1, k),
        right=_build_kdtree(points, mid + 1, end, depth + 1, k),
        axis=axis,
    )
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
type Node struct {
    Point []float64
    Left  *Node
    Right *Node
    Axis  int
}

func BuildKDTree(points [][]float64) *Node {
    // One shared buffer; child calls use index ranges, not copied slices.
    if len(points) == 0 {
        return nil
    }
    k := len(points[0])
    if k == 0 {
        panic("points must have at least one coordinate")
    }
    buf := append([][]float64(nil), points...)
    return buildKDTreeRange(buf, 0, len(buf), 0, k)
}

func buildKDTreeRange(points [][]float64, start, end, depth, k int) *Node {
    n := end - start
    if n <= 0 {
        return nil
    }

    axis := depth % k
    mid := start + n/2
    quickSelect(points, start, end-1, mid, axis)

    return &Node{
        Point: points[mid],
        Left:  buildKDTreeRange(points, start, mid, depth+1, k),
        Right: buildKDTreeRange(points, mid+1, end, depth+1, k),
        Axis:  axis,
    }
}

func quickSelect(points [][]float64, left, right, target, axis int) {
    // Three-way select: stop when target lands in the equal block.
    for left < right {
        pivotIndex := left + rand.Intn(right-left+1)
        lt, gt := partition3(points, left, right, axis, pivotIndex)
        if target < lt {
            right = lt - 1
        } else if target >= gt {
            left = gt
        } else {
            return
        }
    }
}

func partition3(points [][]float64, left, right, axis, pivotIndex int) (int, int) {
    // Rearrange into smaller | equal | larger on this axis.
    // Equal block is points[lt:gt].
    pivotValue := points[pivotIndex][axis]
    lt := left
    i := left
    gt := right + 1
    for i < gt {
        v := points[i][axis]
        if v < pivotValue {
            points[lt], points[i] = points[i], points[lt]
            lt++
            i++
        } else if v > pivotValue {
            // Do not advance i: the swapped-in right point is unexamined.
            gt--
            points[gt], points[i] = points[i], points[gt]
        } else {
            i++
        }
    }
    return lt, gt
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
struct Node {
    std::vector<double> point;
    Node *left = nullptr;
    Node *right = nullptr;
    int axis;

    Node(std::vector<double> pt, int ax) : point(std::move(pt)), axis(ax) {}
};

// Three-way partition into smaller | equal | larger on this axis.
// Equal block is [lt, gt). Any index in that block is a valid median value.
std::pair<std::size_t, std::size_t> partition3(
    std::vector<std::vector<double>>& points,
    std::size_t left, std::size_t right, int axis, std::size_t pivot_index) {
    const double pivot_value = points[pivot_index][axis];
    std::size_t lt = left;
    std::size_t i = left;
    std::size_t gt = right + 1;

    while (i < gt) {
        const double v = points[i][axis];
        if (v < pivot_value) {
            std::swap(points[lt], points[i]);
            ++lt;
            ++i;
        } else if (v > pivot_value) {
            // Do not advance i: the point swapped in from the right
            // is still unexamined.
            --gt;
            std::swap(points[gt], points[i]);
        } else {
            ++i;  // equal: leave it in the middle block
        }
    }
    return {lt, gt};
}

void selectMedian(std::vector<std::vector<double>>& points,
                  std::size_t left, std::size_t right,
                  std::size_t target, int axis) {
    // Expected O(m), including when many points share the same axis value.
    while (left < right) {
        const std::size_t pivot_index =
            left + static_cast<std::size_t>(std::rand()) % (right - left + 1);
        const auto [lt, gt] = partition3(points, left, right, axis, pivot_index);
        if (target < lt) {
            right = lt - 1;   // target lies among the smaller values
        } else if (target >= gt) {
            left = gt;        // target lies among the larger values
        } else {
            return;           // target landed inside the equal block
        }
    }
}

Node* buildKDTreeRange(std::vector<std::vector<double>>& points,
                       std::size_t start, std::size_t end, int depth, int k) {
    if (start >= end) return nullptr;

    const std::size_t n = end - start;
    const int axis = depth % k;
    const std::size_t mid = start + n / 2;
    selectMedian(points, start, end - 1, mid, axis);

    Node* node = new Node(points[mid], axis);
    node->left = buildKDTreeRange(points, start, mid, depth + 1, k);
    node->right = buildKDTreeRange(points, mid + 1, end, depth + 1, k);
    return node;
}

Node* buildKDTree(std::vector<std::vector<double>> points) {
    // Takes the buffer by value so the caller's vector is not rearranged.
    if (points.empty()) return nullptr;
    const int k = static_cast<int>(points[0].size());
    if (k == 0) {
        throw std::invalid_argument("points must have at least one coordinate");
    }
    return buildKDTreeRange(points, 0, points.size(), 0, k);
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

Why is the height about \(\log N\)? Each median split puts roughly half the
points on the left and half on the right. So the number of points still on one
path shrinks like \(N\), then \(N/2\), then \(N/4\), and so on. After \(h\)
halvings you have about \(N / 2^h\) points left. Set that equal to \(1\) and
solve for \(h\):

\[
\frac{N}{2^h} = 1 \quad\Rightarrow\quad 2^h = N \quad\Rightarrow\quad h = \log_2 N.
\]

So \(\log_2 N\) is just the number of times you can cut a pile of \(N\) points
in half before one point remains. That count is the **height** of a
root-to-leaf path when the medians stay balanced. (\(N\) is the number of
stored points. The base \(2\) matches "halve each time"; any other constant
base only changes the unit, not the scaling.)

What does the finished tree promise about coordinates? Only a weak order on
the split axis. If the node stores point \(p\) and split on axis
\(a\), then every point in the left child has
\(x_a \le p_a\), and every point in the right child has \(x_a \ge p_a\).
Equals are allowed on both sides. The builder does **not** promise that the
left side is strictly less than \(p_a\).

That matters for search. Suppose the node splits on \(x\) at \(p_x = 5\), the
query is at \(x = 4.8\), and the best neighbor found so far is distance
\(0.5\) away. The query sits on the left of the plane \(x = 5\), so search
goes left first. The plane is only \(0.2\) away, and \(0.2 < 0.5\), so a
closer point could still sit just to the right of \(x = 5\). Search must
check that far child too.

In code we usually store the best distance as a square,
\(best\_sq = 0.5^2 = 0.25\), so the hot loop never takes a square root. The
plane test must use the same units: compare \(0.2^2 = 0.04\) with
\(0.25\), not \(0.2\) with \(0.25\). That is the same decision as
\(0.2 < 0.5\), because squaring preserves order for nonnegative lengths.
Visit the far side when `diff * diff < best_sq`, where
`diff = query[axis] - node.point[axis]`. A rule that only asks
`query[axis] < node.point[axis]` would skip the far side too early and can
miss the true nearest neighbor.

So the usual search pattern is: follow the near child first, then **prune**
the far child only when the squared distance to the splitting plane is already
at least `best_sq`. A balanced tree therefore does not guarantee a logarithmic
nearest-neighbor query. The geometry decides how many sibling branches still
need a visit. The next subsection tightens that far-child test with a stored
box around each subtree.

{{< mermaid >}}
graph TD
    Root["X < 50?"] -->|Yes| L1["Y < 30?"]
    Root -->|No| R1["Y < 80?"]
    L1 -->|Yes| LL["Leaf: Points A, B"]
    L1 -->|No| LR["Leaf: Points C, D"]
    R1 -->|Yes| RL["Leaf: Points E, F"]
    R1 -->|No| RR["Leaf: Points G, H"]
{{< /mermaid >}}

<p class="figure-note">Figure: each question applies only inside its current
branch. The left and right children can therefore use different local
\(y\)-coordinate medians. A query descends one route first, but exact search
must revisit a sibling whenever that sibling's region could still contain a
closer point.</p>

### Tighter KD pruning with stored MBRs

The single-axis plane test is cheap: one subtract, one multiply, one compare.
It is also weak. It uses only the current split and ignores every constraint
inherited from ancestor splits. A stronger bound stores a tight
**minimum bounding rectangle (MBR)** at each KD node: the axis-aligned box of
the points actually present in that subtree.

{{< reference-figure
  src="kd-tree-plane-vs-mbr.svg"
  label="Plane test versus tight MBR"
  alt="Two plots with the same query, best-neighbor ball, vertical split, and four far points in a distant cluster. Panel A shades the whole far half-plane and marks the short distance from the query to the plane; the ball crosses the plane, so the plane test cannot prune. Panel B draws a tight box only around the far points and marks the longer distance from the query to that box; the box lies outside the ball, so the MBR test can prune."
  caption="Panel A: the plane test only measures the short gap from the query to the split. If that gap is smaller than the best radius, the far child stays open, even when the far points sit in a distant corner. Panel B: a tight MBR bounds only those points. The distance to the box can exceed the best radius, so search can prune the far child."
>}}

**Why the MBR is always at least as strong as the plane.** Think of two
distances from the query to the far child:

1. **Plane distance** \(\lvert\mathrm{diff}\rvert\): how far is the splitting
   plane?
2. **Box distance**: how far is the tight MBR of the points in that far child?

Every far-child point lies on the far side of the plane. The tight MBR is built
only from those points, so the whole box lies on the far side too. Its nearest
face is therefore at the plane or farther away. In symbols,
\(\mathrm{box\_dist}(q,\mathrm{MBR}) \ge \lvert\mathrm{diff}\rvert\) (or with
squares, \(\mathrm{box\_dist}^2 \ge \mathrm{diff}^2\)).

That nesting gives a simple pruning rule:

- If the plane is already farther than the best neighbor, the MBR is at least
  that far, so the MBR test prunes as well.
- If the plane is still close, the MBR may still be far, because the points can
  sit in a distant corner of the far half-plane. Then only the MBR prunes.

{{< reference-figure
  src="kd-tree-prune-implication.svg"
  label="Plane prune implies MBR prune"
  alt="Two cases. Case 1: the best-neighbor ball around the query does not reach the split plane, and a far MBR sits farther right; both plane and MBR decisions say prune. Case 2: the ball crosses the plane so the plane says keep looking, but the same far MBR still lies outside the ball so the MBR still says prune."
  caption="Case 1: when the plane already clears the best ball, the far MBR cannot be closer than the plane, so it prunes too. Case 2: when the ball crosses the plane, the plane test stays open, but the tight box of the far points can still sit outside the ball and prune. That is why the cascade tries the cheap plane test first, then the MBR."
>}}

Store the **tight** box of the points, not the virtual slab carved by the
splitting planes. Both are valid lower-bound regions; the tight box is smaller
and prunes more.

**Cost.** The plane test is \(O(1)\): one subtract, one multiply, one compare
on the current split axis. Adding tight MBRs costs more in three separate
places:

| Cost | Amount | Where it comes from |
|---|---|---|
| Query work per visited node | \(O(k)\) | `box_dist_sq` walks all \(k\) coordinates to measure the query-to-box gap |
| Extra memory per node | \(2k\) floats | each node stores `lo[0..k)` and `hi[0..k)` for its subtree box |
| Extra build work | \(O(nk)\) total | after both children return, the parent unions their boxes coordinate-wise over \(n\) nodes |

Whether that extra \(O(k)\) query work pays for itself is workload-dependent.
There is no universal crossover dimension: the result depends on \(N\),
intrinsic dimension, distribution, metric, query locations, cache behavior,
and implementation. Conventional exact KD-tree methods often lose their
advantage on generic data by a few tens of ambient dimensions, but structured
low-intrinsic-dimensional data can remain searchable at much higher ambient
dimension. Benchmark the plane-only tree, an MBR-enhanced tree, an exact scan,
and the relevant ANN alternatives on the real workload.

**Cascade.** Because the plane test is strictly weaker, run it first. Python's
`and` short-circuits, so a pruned plane never pays for the box test:

```python
if diff * diff < best_sq and box_dist_sq(far, best_sq) < best_sq:
    visit(far)
```

Build the boxes bottom-up on the way out of the same recursion that builds the
tree. The teaching sketch below extends the earlier `Node` with per-axis
`lo` / `hi` bounds. Use `__slots__` in Python so a large tree does not carry a
per-instance `__dict__`.

```python
import math

class Node:
    __slots__ = ("point", "left", "right", "axis", "lo", "hi")

    def __init__(self, point, left, right, axis, lo, hi):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis
        self.lo = lo  # MBR minimum, one value per dimension
        self.hi = hi  # MBR maximum, one value per dimension


def _build_kdtree(points, start, end, depth, k):
    n = end - start
    if n <= 0:
        return None

    axis = depth % k
    mid = start + n // 2
    _select(points, start, end - 1, mid, axis)

    left = _build_kdtree(points, start, mid, depth + 1, k)
    right = _build_kdtree(points, mid + 1, end, depth + 1, k)

    # Tight MBR: this point, unioned with both child boxes.
    p = points[mid]
    lo = list(p)
    hi = list(p)
    for child in (left, right):
        if child is None:
            continue
        for i in range(k):
            if child.lo[i] < lo[i]:
                lo[i] = child.lo[i]
            if child.hi[i] > hi[i]:
                hi[i] = child.hi[i]

    return Node(p, left, right, axis, lo, hi)


def _sqdist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def nearest(root, query):
    if root is None:
        return None, math.inf

    best_point = None
    best_sq = math.inf

    def box_dist_sq(node, limit):
        # Squared distance from query to node's MBR. Stop early if hopeless.
        total = 0.0
        lo, hi = node.lo, node.hi
        for i, q in enumerate(query):
            if q < lo[i]:
                d = lo[i] - q
            elif q > hi[i]:
                d = q - hi[i]
            else:
                continue
            total += d * d
            if total >= limit:
                return total
        return total

    def visit(node):
        nonlocal best_point, best_sq

        d = _sqdist(node.point, query)
        if d < best_sq:
            best_point, best_sq = node.point, d

        axis = node.axis
        diff = query[axis] - node.point[axis]
        near, far = (
            (node.left, node.right) if diff < 0 else (node.right, node.left)
        )

        if near is not None:
            visit(near)

        if far is not None:
            # Plane test first; MBR test runs only if the plane did not prune.
            if diff * diff < best_sq and box_dist_sq(far, best_sq) < best_sq:
                visit(far)

    visit(root)
    return best_point, math.sqrt(best_sq)
```

Both tests use strict `<` for one-nearest-neighbor search so a box that can
only *tie* the current best is not expanded. For a closed range query ("all
points within radius \(r\)"), use `<=` instead.

Stored MBRs also unlock bounds the plane test cannot express. If the farthest
corner of a box already lies inside the query ball, every point in that
subtree qualifies and none needs an individual test. Priority-queue
(best-first) search can expand the closest box next and stop early with an
approximation guarantee. Those are optional elaborations; the cascade above is
the core upgrade.

This raises the ceiling for low-dimensional spatial data. It does not remove
it. As dimension grows, MBRs tend to overlap more, `box_dist_sq` stays near
zero for almost every child, and the traversal approaches a full scan with
extra bookkeeping. Embedding search at hundreds of dimensions needs approximate
methods such as HNSW or IVF-PQ, not a tighter KD bound.

{{< reference-figure
  src="kd-tree-mbr-high-d-limit.svg"
  label="MBR pruning ceiling"
  alt="Two panels. Left: in low dimension a query ball and three separated child boxes; two near boxes are visited and one distant box is pruned. Right: a high-dimension cartoon where four large overlapping boxes all touch the query region, labeled box-dist approximately zero, with the outcome visit almost every child."
  caption="Left: when child boxes sit apart, a far box can clear the best ball and be pruned. Right: a teaching cartoon of high dimension, not a literal embedding plot. Boxes swell and overlap, so the query-to-box distance stays near zero for almost every child. Search visits nearly everything and still pays for the MBR tests."
>}}

**Why do the boxes overlap more as dimension grows?** An MBR is axis-aligned:
for every coordinate it stores only a min and a max. In two dimensions, a
local cluster of points can sit in a small rectangle, and a second cluster can
sit in another rectangle that does not touch the first. Empty corners between
them give the query ball room to miss a whole child.

In higher dimension that picture breaks for a simple counting reason. A point
has \(k\) coordinates. Two points that look "nearby" in the data can still
differ a little on many of those coordinates. The axis-aligned box that covers
even a modest group must therefore stretch almost the full data range on many
axes at once. Stretch enough axes and the box becomes huge. Two huge boxes
inside the same bounded domain have little room to avoid each other, so they
overlap.

Once most child boxes cover most of the space, the query sits inside or beside
nearly every box. Then `box_dist_sq` is near zero for almost every child, the
prune test almost never fires, and search pays for the MBR arithmetic while
still visiting nearly the whole tree. Data with low **intrinsic** dimension
(points that really live on a thin surface inside the high-dimensional room)
can still keep small boxes. Generic full-dimensional scatter usually cannot.

{{< panel "definition" >}}
**KD-tree is not prefix tree.** A KD-tree divides geometric coordinates. A
prefix tree, also called a trie or radix tree, divides discrete keys by their
prefixes. Prefix trees support exact or longest-prefix matching. They do not
usually support spatial nearest-neighbor search.
{{< /panel >}}

### R-trees: group objects inside bounding regions

A KD-tree repeatedly divides the coordinate space. An **R-tree** groups nearby
objects and stores a bounding region for each group.

The name for this low-dimensional bounding box is **minimum bounding rectangle
(MBR)**. In two dimensions, an MBR stores
\((x_{\min},y_{\min})\) and \((x_{\max},y_{\max})\). In \(d\) dimensions, it
stores one minimum and one maximum for each coordinate.

- **Leaf nodes** contain stored objects or references to them.
- **Internal nodes** contain MBRs that enclose their descendants.

Sibling MBRs are not required to be disjoint. If a query region intersects two
sibling boxes, exact search must descend into both. More dimension can increase
overlap on some datasets, but the amount is distribution- and
construction-dependent.

{{< mermaid >}}
graph TD
    Q("Query Point") --> Check{Ball overlaps?}
    Check -->|Yes| A["MBR A (Branch 1)"]
    Check -->|Yes| B["MBR B (Branch 2)"]
    A --> S1["Must Search Branch 1"]
    B --> S2["Must Search Branch 2"]
    style A fill:#ffcccc,stroke:#333
    style B fill:#ffcccc,stroke:#333
    style S1 fill:#ff9999,stroke:#333
    style S2 fill:#ff9999,stroke:#333
{{< /mermaid >}}

<p class="figure-note">Figure: the query ball overlaps both MBR A and MBR B.
The ball-versus-box test alone cannot discard either branch. Exact search must
inspect both. The red styling reinforces the duplicate work. The two labeled
yes-branches carry the same meaning without color.</p>

An R-tree does not use a KD splitting plane. Eligibility is ball-versus-box:
keep a child when the query ball can still touch that child's MBR. The helper
below computes the squared distance from the query to the box and compares it
to a **squared** radius, so the caller never takes a square root only to square
again. The name is `ball_overlaps`, not `intersects`, because this is not a
box-box test.

For one-nearest-neighbor search the comparison is strict (`<`): a box that can
only *tie* the current best is not expanded. For a closed range query ("all
points within radius \(r\)"), use `<=` so a hit exactly at distance \(r\) is
kept. The early exit matters in higher dimensions: most rejected boxes fail
after a few coordinates.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
class MBR:
    def __init__(self, min_coords, max_coords):
        self.min = min_coords  # [x_min, y_min, ...]
        self.max = max_coords  # [x_max, y_max, ...]

    def ball_overlaps(self, query_point, radius_sq):
        """True if any part of this box lies closer than sqrt(radius_sq).

        Takes the squared radius. Strict `<` suits nearest-neighbor search.
        Use `<=` instead for a closed range query.
        """
        total = 0.0
        for i, q in enumerate(query_point):
            if q < self.min[i]:
                d = self.min[i] - q
            elif q > self.max[i]:
                d = q - self.max[i]
            else:
                continue  # inside this slab: contributes nothing
            total += d * d
            if total >= radius_sq:
                return False  # already too far
        return total < radius_sq
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
type MBR struct {
    Min []float64
    Max []float64
}

// BallOverlaps reports whether the query ball can touch this box.
// radiusSq is the squared radius. Strict < suits nearest-neighbor search.
func (m MBR) BallOverlaps(query []float64, radiusSq float64) bool {
    total := 0.0
    for i, q := range query {
        var d float64
        if q < m.Min[i] {
            d = m.Min[i] - q
        } else if q > m.Max[i] {
            d = q - m.Max[i]
        } else {
            continue // inside this slab
        }
        total += d * d
        if total >= radiusSq {
            return false
        }
    }
    return total < radiusSq
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
struct MBR {
    std::vector<double> min_coords;
    std::vector<double> max_coords;

    // Ball-versus-box test. radius_sq is squared. Strict < suits NN search.
    bool ball_overlaps(const std::vector<double>& query, double radius_sq) const {
        double total = 0.0;
        for (size_t i = 0; i < query.size(); ++i) {
            double d = 0.0;
            if (query[i] < min_coords[i]) {
                d = min_coords[i] - query[i];
            } else if (query[i] > max_coords[i]) {
                d = query[i] - max_coords[i];
            } else {
                continue;  // inside this slab
            }
            total += d * d;
            if (total >= radius_sq) {
                return false;
            }
        }
        return total < radius_sq;
    }
};
```
{{< /code-tab >}}
{{< /code-tabs >}}

This R-tree test is the same geometric lower bound used in the KD MBR
cascade above. The difference is control flow: an R-tree may have to apply it
to several overlapping siblings, with no single parent plane to try first.

{{< mermaid >}}
graph TD
    Root["Root (MBR Global)"] --> NodeA["Node A (MBR 1)"]
    Root --> NodeB["Node B (MBR 2)"]
    NodeA --> Obj1("Object 1")
    NodeA --> Obj2("Object 2")
    NodeB --> Obj3("Object 3")
    NodeB --> Obj4("Object 4")
    style Root fill:#f9f,stroke:#333,stroke-width:2px
    style NodeA fill:#ccf,stroke:#333,stroke-width:2px
    style NodeB fill:#ccf,stroke:#333,stroke-width:2px
{{< /mermaid >}}

<p class="figure-note">Figure: an R-tree stores one enclosing MBR at the root,
one MBR per child group, and the objects below those groups. The hierarchy
alone does not show overlap; the previous diagram shows why overlapping sibling
regions can force multiple descents.</p>

### How loss of pruning changes query cost

**Pruning** means discarding a branch because a lower bound proves that the
branch cannot contain a better answer. Suppose every point in a region is at
least 10 units away. If the current best point is 5 units away, the search can
skip that region.

{{< panel "definition" >}}
**Where the logarithm comes from.** Let \(B\) be a balanced tree's branching
factor and \(h\) its height. A node at depth \(h\) represents about
\(N/B^h\) points. Setting \(N/B^h=1\) gives \(h=\log_B N\).

Height is not query cost. If the search frontier stays bounded to a constant
number of nodes per level, a query can inspect \(O(\log N)\) nodes. If it must
recurse into all \(B\) children at every level, the recurrence becomes
\(T(h)=B\,T(h-1)+O(1)\). This recurrence visits \(O(B^h)=O(N)\) nodes. The
failure is loss of pruning, not an unbalanced tree.
{{< /panel >}}

### Classical high-dimensional alternatives: LSH and projection forests

**Locality-sensitive hashing (LSH)** is a family of randomized indexes whose
collision probability reflects a chosen similarity measure. In random
hyperplane LSH for angular similarity, a hyperplane divides space into two
half-spaces. Nearby directions are more likely than distant directions to
receive the same side-of-plane bit. Multiple bits form a bucket key, and
multiple tables raise the chance of recovering a neighbor.

[Charikar's random-hyperplane
construction](https://doi.org/10.1145/509907.509965) is specific to angular
similarity. The
[Johnson-Lindenstrauss result](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)
is a related random-projection result about approximately preserving pairwise
Euclidean distances. It is useful background, but it does not define all LSH
families.

LSH exposes its own recall-space-time tradeoff through the number of tables,
bits, and probes. It remains useful in workloads that match its guarantees; a
graph method does not dominate every LSH configuration on every dataset.

**Random-projection forests**, including
[Annoy](https://github.com/spotify/annoy), build several trees with randomized
hyperplane splits and search across multiple leaves. Annoy in particular uses
an immutable, memory-mapped index design that can be operationally attractive.
Random-projection forests occupy a different point in the build-time, update,
memory, recall, and latency design space from HNSW and ScaNN.

## The challenge: several effects share one name

The **curse of dimensionality** does not mean that one dimension threshold
breaks all trees. The term describes a family of effects. Their strength
depends on the data distribution, metric, sample size, and index.

A B-tree is not the direct answer because it indexes ordered keys. A
multidimensional nearest-neighbor relation does not naturally collapse into
one total order that preserves every neighborhood. KD-trees and R-trees keep
the geometry, but their pruning bounds can become less selective.

### Volume moves away from one chosen center

Consider a unit hypercube with \(d\) coordinates. Its volume is \(1^d=1\). A
centered inner cube whose side length is \(0.9\) occupies the fraction

\[
0.9^d.
\]

That fraction is \(0.81\) in two dimensions and approximately \(0.000027\) in
100 dimensions. Equivalently, under a uniform distribution on this cube, most
probability mass lies outside that particular centered inner cube.

This calculation does **not** prove that every embedding lies on a cube's
surface. It demonstrates how quickly a fixed central region can lose volume as
dimension grows.

### Distances can lose relative contrast

Euclidean squared distance adds one nonnegative contribution per coordinate.
When many coordinate contributions behave similarly and weakly depend on one
another, their sum can have small relative variation compared with its mean.
The nearest and farthest sampled distances can then become less distinct.

Aggarwal, Hinneburg, and Keim analyze this effect across
[high-dimensional distance
metrics](https://doi.org/10.1007/3-540-44503-X_27). The conclusion is
distribution- and metric-dependent. "All points become equally distant" is an
intuition for narrowing contrast, not a literal equality.

The dice analogy needs a clear limit. One die has a wide range relative to one
roll. The sum of 1,000 independent dice clusters more tightly around its mean.
A vector distance is not a die roll. Both examples show that many
contributions can reduce relative variation.

### Intrinsic structure can rescue search

The stored coordinate count is the **ambient dimension**. The **intrinsic
dimension** describes how many effective degrees of freedom the dataset uses
locally or globally. Structured data can occupy a much smaller or more
navigable subset of the ambient space. ANN methods exploit that structure;
they do not repeal the underlying geometry.

High-dimensional neighbor relations can also develop **hubs**: points that
appear in many other points' neighbor lists. The
[hubness study by Radovanović, Nanopoulos, and
Ivanović](https://jmlr.org/papers/v11/radovanovic10a.html) analyzes this
phenomenon. Hubness is something to measure on the actual collection, not a
reason to assume one universal correction.

### A reproducible distance-contrast simulation

The next figure asks one precise question. For each displayed dimension, draw
2,000 independent Gaussian vectors. Normalize each vector to unit length. Use
the first vector as the reference. Measure its Euclidean distance to the other
1,999 vectors. Then divide the nearest distance by the farthest distance.

{{< reference-figure
  src="curse_of_dimensionality.png"
  label="Distance contrast in a synthetic high-dimensional sample"
  alt="A line chart plots a distance ratio for 1,999 random unit vectors and one reference point. The ratio compares the nearest distance with the farthest distance. It rises from about 0.24 at dimension 10 to about 0.91 at dimension 2,000. Near and far distances have less contrast in this simulation."
  caption="**Teaching simulation:** the nearest-to-farthest ratio rises toward one in this seeded unit-sphere protocol. The figure supports one distribution-specific lesson: relative distance contrast narrows here. It does not show that every real embedding space becomes useless or that an ANN index must fail. Reproduce it with [reproduce.py](reproduce.py); exact values and hashes are in the [figure receipt](fig-vector-search-teaching.receipt.json), with [provenance](provenance.json)."
>}}

The page-owned [`reproduce.py`](reproduce.py) is the authoritative generator.
The Python excerpt below matches its numeric protocol. The Go and C++ ports
implement the same sampling steps, but their standard-library random-number
generators do not produce NumPy's byte-identical stream. They are teaching
ports, not sources for the committed receipt.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import numpy as np

dimensions = [10, 25, 50, 100, 250, 500, 1000, 2000]
point_count = 2000
rng = np.random.default_rng(42)

for dimension in dimensions:
    points = rng.standard_normal((point_count, dimension))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    distances = np.linalg.norm(points[1:] - points[0], axis=1)
    ratio = float(distances.min() / distances.max())
    print(dimension, ratio)
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
package main

import (
	"fmt"
	"math"
	"math/rand"
)

func calculateRatio(rng *rand.Rand, dim, numPoints int) float64 {
	points := make([][]float64, numPoints)
	for i := range points {
		points[i] = make([]float64, dim)
		var norm float64
		for j := 0; j < dim; j++ {
			v := rng.NormFloat64()
			points[i][j] = v
			norm += v * v
		}
		norm = math.Sqrt(norm)
		for j := 0; j < dim; j++ {
			points[i][j] /= norm
		}
	}

	minDist, maxDist := math.MaxFloat64, 0.0

	// Compare point 0 to all others
	for i := 1; i < numPoints; i++ {
		var distSq float64
		for k := 0; k < dim; k++ {
			diff := points[0][k] - points[i][k]
			distSq += diff * diff
		}
		dist := math.Sqrt(distSq)
		if dist < minDist {
			minDist = dist
		}
		if dist > maxDist {
			maxDist = dist
		}
	}

	return minDist / maxDist
}

func main() {
	rng := rand.New(rand.NewSource(42))
	dimensions := []int{10, 25, 50, 100, 250, 500, 1000, 2000}
	for _, dim := range dimensions {
		ratio := calculateRatio(rng, dim, 2000)
		fmt.Printf("%d,%.4f\n", dim, ratio)
	}
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

double calculate_ratio(
    std::mt19937& gen,
    int dim,
    int num_points = 2000
) {
    std::vector<std::vector<double>> points(num_points, std::vector<double>(dim));
    std::normal_distribution<> d(0, 1);

    for (int i = 0; i < num_points; ++i) {
        double norm_sq = 0.0;
        for (int j = 0; j < dim; ++j) {
            points[i][j] = d(gen);
            norm_sq += points[i][j] * points[i][j];
        }
        double norm = std::sqrt(norm_sq);
        for (int j = 0; j < dim; ++j) {
            points[i][j] /= norm;
        }
    }

    double min_dist = 1e9;
    double max_dist = 0.0;

    // Compare point 0 to all others
    for (int i = 1; i < num_points; ++i) {
        double dist_sq = 0.0;
        for (int k = 0; k < dim; ++k) {
            double diff = points[0][k] - points[i][k];
            dist_sq += diff * diff;
        }
        double dist = std::sqrt(dist_sq);
        min_dist = std::min(min_dist, dist);
        max_dist = std::max(max_dist, dist);
    }
    return min_dist / max_dist;
}

int main() {
    std::mt19937 gen(42);
    const std::vector<int> dimensions = {
        10, 25, 50, 100, 250, 500, 1000, 2000
    };
    std::cout << "dimension,ratio\n";
    for (int dim : dimensions) {
        double ratio = calculate_ratio(gen, dim);
        std::cout << dim << "," << std::fixed << std::setprecision(4) << ratio << "\n";
    }
    return 0;
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

The figure shows why a shortcut can be attractive. It does not justify one.
An exact baseline and an accuracy contract must come first. An ANN index can
then trade exact-neighbor recall for a measured reduction in work.

---

## Distance metrics: define what "near" means

Let \(q=(q_1,\ldots,q_d)\) be a query vector and
\(x=(x_1,\ldots,x_d)\) one stored vector. The integer \(d\) is the number of
coordinates. The subscript \(j\) identifies one coordinate, from \(1\) through
\(d\).

### Euclidean distance: endpoint separation

**Euclidean distance**, also called L2 distance, generalizes straight-line
distance:

\[
d_{\mathrm{L2}}(q,x)
=\sqrt{\sum_{j=1}^{d}(q_j-x_j)^2}.
\]

Smaller Euclidean distance means a nearer endpoint. Both direction and vector
length affect the result.

### Inner product: coordinate alignment with length

The **inner product**, also called the dot product, multiplies corresponding
coordinates and adds the products:

\[
q\cdot x=\sum_{j=1}^{d}q_jx_j.
\]

Larger is better for maximum inner-product search (MIPS). Vector length
matters: a long vector can receive a large inner product without being the
closest direction.

### Cosine similarity: directional agreement

The L2 norm is a vector's Euclidean length:

\[
\lVert x\rVert_2=\sqrt{\sum_{j=1}^{d}x_j^2}.
\]

For two nonzero vectors, {{< refterm "cosine-similarity" "cosine similarity" >}}
divides the inner product by both lengths:

\[
\operatorname{cosine}(q,x)
=\frac{q\cdot x}{\lVert q\rVert_2\lVert x\rVert_2}.
\]

The result lies from \(-1\) to \(1\). It is undefined when either vector has
zero length. Cosine similarity measures geometric direction under this
representation; it is not a probability that an item is relevant.

### Baseline: linear exact nearest-neighbor search

An exact scan evaluates the chosen metric against all \(N\) stored vectors.
With \(d\) coordinates per vector, scoring costs \(O(Nd)\). Returning the best
one requires one running minimum or maximum. Returning the best \(k\) also
needs a top-\(k\) selection structure or a sort.

The three snippets below show exact one-nearest-neighbor search with Euclidean
distance. They omit production concerns such as identifiers, dimension checks,
batching, vectorized kernels, and tie rules.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import numpy as np

def linear_search(query, dataset):
    best_dist = float('inf')
    best_item = None

    for item in dataset:
        dist = np.linalg.norm(query - item)
        if dist < best_dist:
            best_dist = dist
            best_item = item

    return best_item, best_dist
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
func LinearSearch(query []float64, dataset [][]float64) ([]float64, float64) {
	bestDist := math.MaxFloat64
	var bestItem []float64

	for _, item := range dataset {
		dist := EuclideanDistance(query, item)
		if dist < bestDist {
			bestDist = dist
			bestItem = item
		}
	}
	return bestItem, bestDist
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <vector>
#include <limits>
#include <cmath>

std::pair<std::vector<double>, double> linear_search(const std::vector<double>& query, const std::vector<std::vector<double>>& dataset) {
    double best_dist = std::numeric_limits<double>::max();
    std::vector<double> best_item;

    for (const auto& item : dataset) {
        double dist = euclidean_distance(query, item);
        if (dist < best_dist) {
            best_dist = dist;
            best_item = item;
        }
    }
    return {best_item, best_dist};
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

### Check the three teaching candidates

For the query \(q=(0.80,0.60)\), the three rules give:

| Candidate | Euclidean distance, smaller is better | Inner product, larger is better | Cosine similarity, larger is better |
|---|---:|---:|---:|
| related weather | \(0.078\) | \(0.996\) | \(0.997\) |
| straight up | \(0.894\) | \(0.600\) | \(0.600\) |
| unrelated car problem | \(1.980\) | \(-0.960\) | \(-0.960\) |

All three metrics rank this particular toy set in the same order. That
agreement is not automatic for vectors with different lengths.

### Cosine implementations in three languages

Each function below computes cosine similarity directly. A production version
must also reject unequal dimensions and specify its response to zero vectors.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import numpy as np

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return float("nan")
    return dot_product / (norm_v1 * norm_v2)
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
func CosineSimilarity(v1, v2 []float64) float64 {
	var dot, norm1, norm2 float64
	for i := range v1 {
		dot += v1[i] * v2[i]
		norm1 += v1[i] * v1[i]
		norm2 += v2[i] * v2[i]
	}
	if norm1 == 0 || norm2 == 0 {
		return math.NaN()
	}
	return dot / (math.Sqrt(norm1) * math.Sqrt(norm2))
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <vector>
#include <cmath>
#include <limits>

double cosine_similarity(const std::vector<double>& v1, const std::vector<double>& v2) {
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 == 0 || norm2 == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

{{< panel "definition" >}}
**Metric reductions without the skipped algebra**

If both \(q\) and \(x\) have length \(1\), then

\[
\operatorname{cosine}(q,x)=q\cdot x
\quad\text{and}\quad
\lVert q-x\rVert_2^2=2-2(q\cdot x).
\]

For the same unit vectors, increasing inner product, increasing cosine
similarity, and decreasing Euclidean distance therefore produce the same
ranking. Their numeric scores are still different.

For unnormalized maximum inner-product search, choose
\(R\ge\max_x\lVert x\rVert_2\). Map each stored vector to

\[
x'=\left[x;\sqrt{R^2-\lVert x\rVert_2^2}\right]
\]

and map the query to \(q'=[q;0]\). Every \(x'\) has norm \(R\), so

\[
\lVert q'-x'\rVert_2^2
=\lVert q\rVert_2^2+R^2-2(q\cdot x).
\]

The first two terms do not change across stored items. Minimizing this
Euclidean distance therefore maximizes \(q\cdot x\). This reduction requires
the declared bound \(R\); it is not the same as silently treating an
unnormalized dot product as cosine.
{{< /panel >}}

## The ANN contract: what approximation is allowed to miss

An **approximate nearest-neighbor (ANN)** index avoids some of the work in an
exact scan. It might skip graph branches, search only selected partitions, or
score compressed representations before reranking. The shortcut can omit a
true vector neighbor. Therefore, reports must show speed and accuracy together.

Let \(T_k(q)\) be the set of exact top-\(k\) item identifiers for query \(q\).
Let \(A_k(q)\) be the \(k\) identifiers returned by approximate search. With a
fixed metric, eligible collection, filter, snapshot, and tie rule, exact-neighbor
recall at \(k\) is

\[
\operatorname{Recall@}k(q)
=\frac{\left|T_k(q)\cap A_k(q)\right|}{k}.
\]

The vertical bars count identifiers in the intersection. If an approximate
top 10 contains 9 of the exact top 10, its recall at 10 is \(9/10=0.9\) for
that query. A benchmark normally averages the per-query value over a declared
query set.

{{< panel "warning" >}}
**Exact-neighbor recall is not user relevance.** It asks whether the ANN index
reproduced the exact vector ranking. It does not ask whether either ranking
answers the user's question. Application evaluation needs separate
{{< refterm "relevance-judgments" "relevance judgments" >}} and
{{< refterm "retrieval-ranking-metrics" "ranked-list metrics" >}}.
{{< /panel >}}

---

## Graph-based indexing: HNSW

**HNSW** stands for **Hierarchical Navigable Small World**. It is a widely
deployed graph-based ANN design. HNSW works well when the graph and vector
payloads can remain in memory. Malkov and Yashunin describe the original
algorithm <a id="cite-malkov-hnsw"></a>[[malkov-hnsw]](#ref-malkov-hnsw).

These terms define the graph:

* A **node** represents one indexed vector.
* An **edge** records that the construction algorithm chose another node as a useful neighbor.
* A **proximity graph** connects nodes that are near under the index's distance function.
* **Navigable** means a local routing rule can often move through that graph toward the query without inspecting every node.

HNSW does not place vectors into a binary search tree. It builds several increasingly sparse graph layers and routes through their edges.

### Small-world navigation

A small-world graph combines mostly local connections with some longer-range
connections. The longer connections help to keep routes short. In a social
network, each person knows a small fraction of all people. A short chain of
acquaintances can still cross the network.

That analogy is not a complexity proof. HNSW's strong search behavior is empirical and depends on the data distribution, metric, graph construction, and search budget. Its worst case can still be poor.

### The layered road system

HNSW borrows its hierarchy from skip lists:

* **Upper layers** contain only a sparse sample of nodes. Their edges act like long-range roads.
* **Middle layers** contain more nodes and shorter-range connections.
* **Layer 0** contains every indexed node. It supplies the final candidate neighborhood.

{{< mermaid >}}
graph TD
    subgraph L1 ["Layer 1: Interstates (Sparse)"]
        E((Entry)) -- 100km --> H((Hub))
        H -- 200km --> F((Far))
    end

    subgraph L0 ["Layer 0: Local Roads"]
        H_L0((Hub)) -- 5km --> N1((Node A))
        N1 -- 1km --> T((Target))
    end

    H -. "Exit Highway" .-> H_L0

    style T fill:#f96,stroke:#333,stroke-width:4px
    style H fill:#f9f,stroke:#333
    style H_L0 fill:#f9f,stroke:#333
{{< /mermaid >}}

*Figure reading guide: search begins in the sparse upper layer and takes a
long hop toward the query region. It then descends to the denser layer for
shorter hops. The kilometer labels are an analogy, not measured vector
distances.*

### Search: greedy descent, then bounded best-first expansion

At an upper layer, the search starts from an entry point. It measures the
query-to-node distance for each neighbor and moves to a closer neighbor. This
process stops when no neighbor improves the current position. The search then
uses that position as the entry point for the next layer.

Layer 0 needs a wider search. A single locally best route can stop at a
**local minimum**. At that node, all immediate neighbors look worse, although
a better region exists elsewhere. HNSW therefore maintains more than one live
possibility.

{{< mermaid >}}
graph TD
    Q((Query Peak))

    subgraph Step1 ["Step 1: Check Neighbors"]
        Current((Start)) --> N1((Node 1))
        Current --> N2((Node 2))
        Current --> N3((Node 3))
    end

    subgraph Step2 ["Step 2: Move to Best & Expand"]
        N3 --> N4((Node 4))
        N3 --> N5((Node 5))
    end

    %% Distances to Query (Dotted Lines)
    N1 -. "10m" .-> Q
    N2 -. "50m" .-> Q
    N3 -. "5m (Best)" .-> Q
    N4 -. "4m" .-> Q
    N5 -. "1m (Best)" .-> Q

    style N3 fill:#f9f,stroke:#333,stroke-width:4px
    style N5 fill:#f9f,stroke:#333,stroke-width:4px
{{< /mermaid >}}

*Figure reading guide: the dotted lines are query-distance evaluations. Node 3
is the closest known candidate, so the algorithm expands it. The expansion
finds Nodes 4 and 5. The best known distance improves from 5 to 1.*

The paper's `SEARCH-LAYER` procedure is easiest to understand with two queues:

1. Put the entry point into a **candidate frontier** \(C\), ordered so the nearest unexplored candidate comes out first.
2. Put the entry point into a bounded **working result set** \(W\), ordered so its worst retained candidate is easy to inspect.
3. Remove the nearest candidate \(c\) from \(C\), then evaluate each unvisited graph neighbor \(e\) of \(c\).
4. If \(W\) is not full, or \(e\) is closer than the worst item in \(W\), add \(e\) to both queues.
5. If \(W\) exceeds the configured width \(ef\), remove its worst item.
6. Stop once the nearest unexplored candidate in \(C\) is farther than the worst retained item in a full \(W\).
7. Return the best \(k\) items from \(W\).

Implementations differ in queue layout, deletion handling, prefetching, and
termination details. However, the two queue roles remain important. \(C\)
identifies the next candidate to expand. \(W\) contains the best retained
candidates.

{{< panel "info" >}}
**What `ef` actually controls**

`ef_search` controls the search width. It does not specify a number of threads
or parallel workers. A larger value lets the search retain and usually visit
more alternatives. This change usually improves recall. It also increases
distance evaluations, memory traffic, and latency. The requested neighbor
count \(k\) normally requires `ef_search >= k`.
{{< /panel >}}

### Index construction

When a new vector arrives, HNSW:

1. samples the new node's maximum layer;
2. descends greedily from the current top entry point to the new node's top layer;
3. searches more broadly at each participating layer using `ef_construction`;
4. selects a bounded, diverse set of neighbors rather than merely the closest duplicates in one direction;
5. adds reciprocal links and prunes adjacency lists that exceed their caps.

That last neighbor-selection heuristic is a major part of the algorithm. Random levels alone do not create a useful proximity graph.

### Layer assignment in code

The following snippets implement only maximum-layer sampling. They are not complete HNSW indexes. A common parameterization samples \(U\) uniformly from \((0,1)\) and computes

\[
\operatorname{level}=\left\lfloor-\ln(U)\,m_L\right\rfloor.
\]

The examples choose \(m_L=1/\ln(M)\). The next section derives this common
implementation choice.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import random
import math

class HNSWNode:
    def __init__(self, value, level):
        self.value = value
        # Neighbors list for each layer from 0 to level
        self.neighbors = [[] for _ in range(level + 1)]

class HNSWGraph:
    def __init__(self, M=16, max_level=16):
        self.M = M
        self.max_level = max_level
        # The normalization factor
        self.m_L = 1.0 / math.log(M)

    def random_level(self):
        # HNSW layer assignment follows an exponential distribution.
        # This ensures the probability of reaching layer L decays by 1/M at each step.
        f = random.uniform(0, 1)
        # Avoid log(0) edge case
        if f == 0.0:
            f = 1e-10

        level = int(-math.log(f) * self.m_L)
        return min(level, self.max_level)
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
package main

import (
	"math"
	"math/rand"
)

type HNSWNode struct {
	Value     int
	Neighbors [][]int // Indexes of neighbors at each layer
}

type HNSWGraph struct {
	M        int
	MaxLevel int
	mL       float64
	rng      *rand.Rand
}

func NewHNSWGraph(m int, maxLevel int, seed int64) *HNSWGraph {
	return &HNSWGraph{
		M:        m,
		MaxLevel: maxLevel,
		mL:       1.0 / math.Log(float64(m)),
		rng:      rand.New(rand.NewSource(seed)),
	}
}

func (g *HNSWGraph) RandomLevel() int {
	// Exponential distribution scaled by 1/ln(M)
	f := g.rng.Float64()
	if f == 0.0 {
		f = 1e-10
	}

	level := int(-math.Log(f) * g.mL)

	if level > g.MaxLevel {
		return g.MaxLevel
	}
	return level
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

struct HNSWNode {
    int value;
    std::vector<std::vector<int>> neighbors; // Neighbors for each layer

    HNSWNode(int v, int level) : value(v), neighbors(level + 1) {}
};

class HNSWGraph {
    int M;
    int max_level;
    double m_L;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;

public:
    HNSWGraph(int m, int max_lvl)
        : M(m), max_level(max_lvl), m_L(1.0 / std::log(m)), gen(42), dist(0.0, 1.0) {}

    int randomLevel() {
        // Sample from uniform(0,1), take -ln(), scale by m_L
        double f = dist(gen);
        // Avoid log(0)
        if (f == 0.0) f = 1e-10;

        int lvl = static_cast<int>(-std::log(f) * m_L);
        return std::min(lvl, max_level);
    }
};
```
{{< /code-tab >}}
{{< /code-tabs >}}

### Purpose of random levels

A batch algorithm could select geometry-aware landmarks from the complete
dataset. However, this global coordination makes online insertion and
maintenance more complicated. Independent random level assignment is
inexpensive and requires no tree rotations. It also gives every inserted node
the same chance of reaching a sparse upper layer.

This randomness affects **index construction**. Two builds can differ because
of their random seeds, insertion orders, or concurrency schedules. An
unchanged immutable index does not necessarily return different results for
the same query. The section on non-determinism separates build reproducibility
from query-time behavior.

> **Clarification: "Existing" in multiple layers**
> Nodes are not duplicated. "Existing in Layer $L$" means that the node has
> links at that layer. A top-layer node also has neighbor sets for all lower
> layers. Thus, one indexed node can act as a multi-level hub.

{{< mermaid >}}
graph TD
    subgraph L2 ["Layer 2 (Express)"]
        A2((Node A)) --- B2((Node B))
    end

    subgraph L1 ["Layer 1 (Regional)"]
        A1((Node A)) --- B1((Node B))
        A1 --- C1((Node C))
        B1 --- D1((Node D))
    end

    subgraph L0 ["Layer 0 (All Nodes)"]
        A0((Node A)) --- B0((Node B))
        A0 --- C0((Node C))
        B0 --- D0((Node D))
        C0 --- E0((Node E))
        D0 --- E0((Node E))
    end

    %% Conceptual links showing it's the same node
    A2 -.-> A1
    A1 -.-> A0
    B2 -.-> B1
    B1 -.-> B0
    C1 -.-> C0
    D1 -.-> D0

    style A2 fill:#f9f,stroke:#333
    style A1 fill:#f9f,stroke:#333
    style A0 fill:#f9f,stroke:#333
{{< /mermaid >}}

*Figure reading guide: A and B are single indexed nodes. They have adjacency
lists at Layers 0, 1, and 2. C and D reach Layer 1. E exists only at Layer 0.
The dotted vertical connectors show identity across layers. They are not
search edges.*

### Derive the layer population

First consider the **uncapped** sampled level. Let
\(U\sim\operatorname{Uniform}(0,1)\). For every non-negative integer layer
\(\ell\),

\[
\begin{aligned}
P(\operatorname{level}\ge \ell)
&=P\left(-\ln(U)m_L\ge \ell\right)\\
&=P\left(U\le e^{-\ell/m_L}\right)\\
&=e^{-\ell/m_L}.
\end{aligned}
\]

If \(m_L=1/\ln M\), then

\[
P(\operatorname{level}\ge \ell)=e^{-\ell\ln M}=M^{-\ell}.
\]

So among \(N\) indexed nodes, the **expected** number present at layer \(\ell\) is

\[
E[N_\ell]=N M^{-\ell}.
\]

The three teaching implementations store
`min(raw_level, max_level)`. If that cap is \(L_{\max}\), the displayed tail
formula remains valid for \(0\le\ell\le L_{\max}\). For
\(\ell>L_{\max}\), the stored-level probability and expected population are
zero.

The expected population becomes approximately one when \(N M^{-\ell}\approx1\), or

\[
\ell\approx\log_M N.
\]

This derivation explains the geometric thinning of the hierarchy. It does **not** prove that \(1/M\) is universally optimal, nor does it prove the entire query takes \(O(\log N)\) time.

{{< panel "warning" >}}
**Two parameters that are easy to conflate**

The graph-connectivity parameter \(M\) differs from the level multiplier
\(m_L\). The original paper relates them with a convenient choice. Many
libraries derive \(m_L\) internally. Neighbor limits also depend on the
implementation. Upper layers commonly use a limit related to \(M\). Layer 0
often uses a larger limit, such as \(2M\). Stored degree can be lower than
either limit.
{{< /panel >}}

### A bounded complexity model

The HNSW paper reports approximately logarithmic empirical scaling for tested datasets. A useful accounting model is:

\[
\begin{aligned}
\text{query work}
&\approx \text{upper-layer distance evaluations}\\
&\quad+\text{Layer-0 expansions}\times\text{neighbors checked per expansion}.
\end{aligned}
\]

Each distance evaluation costs \(O(d)\) for a dense \(d\)-dimensional vector.
Quantization or vector instructions can change the representation and constant
cost. `ef_search` limits a retained working set. It does not give an exact
count of visited nodes or distance evaluations. Duplicate discoveries, degree
variation, deletion markers, filters, and stopping rules also affect the
count.

Therefore:

* \(O(\log N)\) is a useful **empirical description**, not an unconditional worst-case guarantee.
* Increasing \(M\) can improve connectivity and recall, but costs graph memory, build work, and per-expansion checks.
* Increasing `ef_construction` usually improves graph quality, but increases build time.
* Increasing `ef_search` usually improves recall, but increases query work and tail latency.

| Parameter | Used when | What it directly changes | Common tradeoff |
|:--|:--|:--|:--|
| \(M\) | Build and storage | Neighbor-list capacity and graph connectivity | Recall and robustness versus RAM, build time, and distance checks |
| `ef_construction` | Build | Candidate-search breadth while choosing links | Better graph quality versus slower indexing |
| `ef_search` | Query | Breadth of the bounded Layer-0 search | Higher recall versus latency and central-processing-unit (CPU) or memory bandwidth |
| \(k\) | Query result | Number of requested neighbors | Must fit within the search candidate budget |

> **Why the search keeps more than one candidate**
>
> A greedy search keeps only the best current candidate. It can stop at a
> **local minimum**, although a better region exists.
>
> Think of a hiker in thick fog. The hiker always takes the steepest uphill
> path. A small hill can stop the hiker because every immediate step goes down.
> The higher summit can still exist across a valley.
>
> Keeping **`ef`** candidates preserves multiple routes. If the best current
> route stops, another retained route can cross the valley. The wider search
> reduces the risk of a local-minimum failure.

{{< reference-figure
  src="local_vs_global.png"
  label="Greedy routing and a wider candidate search"
  alt="A contour landscape contains a shallow basin on the left and a deeper basin on the right. A dashed greedy route retains one candidate and stops in the left basin. A solid wider route crosses the ridge and reaches the deeper basin."
  caption="A conceptual routing landscape. The dashed path retains one locally best continuation. The solid path preserves alternatives long enough to cross a worse intermediate region. The figure teaches a local-minimum failure mode. It is not an HNSW execution trace, measured recall curve, or optimization objective. [reproduce.py](reproduce.py) generated the figure. The [figure receipt](fig-vector-search-teaching.receipt.json) records its hashes."
>}}

HNSW is not the only graph family. Neighborhood Graph and Tree (NGT),
nearest-neighbor descent (NN-Descent/PyNNDescent), Navigating Spreading-out
Graph (NSG), Vamana, and newer systems make different construction, pruning,
storage, and hardware-layout choices. A benchmark compares complete
implementations and tunings, not an abstract algorithm alone. The production
checklist applies this distinction.

---

## Disk-based indexing: DiskANN and Vamana

An in-memory graph can be fast. At very large scale, its memory cost can be
too high. The vector payload alone costs

\[
N\times d\times\text{bytes per coordinate},
\]

before neighbor IDs, allocator overhead, deletion state, and metadata. One
billion 768-dimensional `float32` vectors require about 3.07 TB for the raw
coordinates alone.

Microsoft Research designed **DiskANN** for high-recall ANN with a large
SSD-resident index <a id="cite-diskann-neurips"></a>[[diskann-neurips]](#ref-diskann-neurips).
The design calls its search graph **Vamana**.

### Vamana: a flat graph with long and short links

HNSW creates explicit sparse layers. Vamana uses one graph with local and
longer-range edges. Its degree-bounded **RobustPrune** procedure selects the
candidates that receive the limited edge slots.

Suppose \(p\) is the node whose outgoing neighbors we are selecting. The procedure repeatedly keeps the remaining candidate \(u\) closest to \(p\). It may then remove another candidate \(v\) when

\[
\alpha\,d(u,v)\le d(p,v).
\]

Let \(\alpha\ge1\) be the pruning parameter. After the procedure retains
\(u\), the inequality says that \(v\) is sufficiently closer to \(u\) than to
\(p\) under the \(\alpha\) margin. The construction therefore treats the
direct edge \(p\rightarrow v\) as redundant enough to prune. It does **not**
claim that the complete path \(p\rightarrow u\rightarrow v\) is shorter than
the direct edge. A degree bound \(R\) stops the retained neighbor set from
growing without limit.

For \(\alpha>1\), the condition is stricter than the \(\alpha=1\)
relative-neighborhood rule. An already retained neighbor must cover a candidate
especially well before pruning removes that candidate. This rule can preserve
more long-range connectivity. It can also increase build work and degree
pressure.

{{< mermaid >}}
graph TD
    P(("Center Node P"))
    
    subgraph K1 ["Keep (Different Direction)"]
        C(("Neighbor C"))
    end
    
    subgraph P1 ["Prune (Redundant)"]
        C_prime(("Neighbor C'"))
    end
    
    %% Edges
    P -- "Close (10m)" --> C
    P -. "Far but reachable via C (15m)" .-> C_prime
    C -- "Short hop (6m)" --> C_prime
    
    style P fill:#f9f,stroke:#333
    style C fill:#ccf,stroke:#333
    style C_prime fill:#fcc,stroke:#333,stroke-dasharray: 5 5
{{< /mermaid >}}

*Figure reading guide: \(C'\) lies in nearly the same direction as retained
neighbor \(C\). With the displayed distances, the pruning condition holds only
when \(\alpha\le2.5\); for example, \(\alpha=2\) gives
\(2\times6\le15\). The complete two-hop length is \(10+6=16\), so the figure
does not claim that the two-hop route is shorter than the direct length 15.
Actual pruning uses the metric inequality and degree bound, not a literal
compass-angle test.*

### Arrange the search around SSD behavior

Solid-state drives (SSDs) are much faster than disks with moving heads, but
random reads remain far slower than CPU cache or dynamic random-access memory
(DRAM). DiskANN therefore coordinates several representations:

1. **SSD-resident records** store graph neighborhoods and full-precision vector data in layouts intended to make a node expansion require few reads.
2. **DRAM-resident compressed vectors** provide inexpensive approximate distance estimates for candidate ordering.
3. A **cache** may keep frequently visited graph nodes in memory.
4. Search can issue several SSD reads together so the device has useful parallel work.

DiskANN is not HNSW stored in a file. Its graph construction and record layout
account for the storage medium. Compressed comparisons, caching, beam search,
and input/output (I/O) batching also account for SSD behavior.

DiskANN does not guarantee a fixed memory reduction or fixed latency. Memory,
recall, and tail latency depend on several factors. These include vector
dimension, compression size, graph degree, cache size, SSD, concurrency, and
search width. DiskANN exposes this memory and I/O tradeoff for datasets that
are too expensive to keep in DRAM.

## Partitioning: IVF

**IVF** means **inverted file index**. It reduces the candidate count by
partitioning vectors into lists. Faiss is an influential library that
implements IVF and many related combinations <a id="cite-faiss-paper"></a>[[faiss-paper]](#ref-faiss-paper).

This inverted file differs from the term-to-document inverted index used by
BM25. Both structures provide indirect access to candidate lists. However,
IVF uses learned vector centroids as keys instead of words.

### Coarse quantization and Voronoi cells

Training learns \(K\) centroid vectors, often with k-means on a representative
sample. A centroid is the coordinate-wise center assigned to a cluster. These
centroids produce **Voronoi cells**. Each location belongs to its nearest
centroid under the selected metric.

Index construction assigns each database vector \(x\) to a nearest centroid
\(c(x)\). It appends the vector identifier and payload or compressed code to
that centroid's list. Faiss commonly calls the number of lists `nlist`.

At query time:

1. compare query \(q\) with the coarse centroids;
2. choose the `nprobe` nearest centroids;
3. scan candidates only in those selected lists;
4. keep the best results under the fine distance estimate;
5. optionally re-rank a larger candidate pool with original vectors.

{{< mermaid >}}
graph TD
    Q((Query))
    
    subgraph Coarse ["Step 1: Coarse Quantizer (Find Centroids)"]
        C1{{Centroid 1}}
        C2{{Centroid 2}}
        C3{{Centroid 3}}
    end
    
    subgraph Fine ["Step 2: Fine Search (Scan Buckets)"]
        B1["Bucket 1 (Ignored)"]
        B2["Bucket 2 (Scanned)"]
        B3["Bucket 3 (Ignored)"]
        
        vec1[Vector A]
        vec2[Vector B]
        vec3[Vector C]
    end
    
    %% Edges
    Q -- "Dist: 10" --> C1
    Q -- "Dist: 2 (Closest)" --> C2
    Q -- "Dist: 8" --> C3
    
    C1 -.-> B1
    C2 -.-> B2
    C3 -.-> B3
    
    B2 --- vec1
    B2 --- vec2
    B2 --- vec3
    
    style C2 fill:#f9f,stroke:#333,stroke-width:2px
    style B2 fill:#ccf,stroke:#333,stroke-width:2px
    style Q fill:#ff9,stroke:#333
{{< /mermaid >}}

*Figure reading guide: the coarse stage finds Centroid 2 nearest to the query.
Therefore, the fine stage scans Bucket 2. With `nprobe = 1`, it skips Buckets 1
and 3. A skipped bucket could contain a vector near the cell boundary.*

A larger `nprobe` examines more lists, so it usually improves recall and increases work. Setting `nprobe = nlist` removes the coarse-pruning benefit and approaches a full scan of all lists.

There is no universal law that \(K=\sqrt N\). This expression is a historical
rule of thumb for some regimes. It is not a production requirement. Select
`nlist` and `nprobe` with measurements. List imbalance, training quality,
vector distribution, filters, batch size, hardware, and compression all change
the best values.

IVF can miss a true neighbor at a cell boundary. That neighbor can be in an
unprobed list on the other side of the boundary. Multiple probes reduce this
risk. Multi-assignment during indexing can also reduce boundary misses. It
uses more storage and maintenance work.

---

## Compression: quantization

Candidate pruning selects the vectors to inspect. Quantization reduces the
cost to store and compare each selected vector. It maps a high-precision
representation to a smaller discrete code. Unless stated otherwise, this
mapping is lossy. Different vectors can receive the same or similar code.

### Product Quantization (PQ)

**Product quantization** splits a \(d\)-dimensional vector into \(m\) disjoint
subvectors. It trains one small centroid codebook per subspace. Encoding
replaces each subvector with the integer identifier (ID) of its nearest
codeword.

With 256 codewords per subspace, one codeword ID fits in one byte. For a 384-dimensional `float32` vector divided into 48 eight-dimensional subvectors:

* original vector: \(384\times4=1{,}536\) bytes;
* PQ code: \(48\times1=48\) bytes;
* code-only compression: \(32\times\).

The codebooks and record metadata also consume memory, so 32× is not necessarily the whole index's compression ratio.

PQ can encode original vectors directly. In an **IVF-PQ** or IVFADC design, it is common to encode the residual

\[
r(x)=x-c(x),
\]

because the coarse centroid has already explained part of the vector. Residual coding is an IVF-PQ technique, not a requirement of PQ in every architecture.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import numpy as np
from sklearn.cluster import KMeans

def train_pq(training_vectors, m=48):
    # Pass residuals here for an IVF-PQ design.
    dim = training_vectors.shape[1]
    assert dim % m == 0, "Dimension must be perfectly divisible by m"
    sub_dim = dim // m
    codebooks = []
    
    for i in range(m):
        # Extract sub-vectors for this subspace
        start = i * sub_dim
        end = (i + 1) * sub_dim
        sub_vectors = training_vectors[:, start:end]
        
        # Train k-means (256 centroids = 1 byte)
        # Note: At scale, PQ codebooks are trained with sampling 
        # + minibatch k-means / graphics-processing-unit (GPU) implementations.
        kmeans = KMeans(n_clusters=256, random_state=42, n_init="auto")
        kmeans.fit(sub_vectors)
        codebooks.append(kmeans.cluster_centers_)
        
    return codebooks

def encode_pq(vector, codebooks):
    m = len(codebooks)
    sub_dim = len(vector) // m
    encoded = []
    
    for i in range(m):
        start = i * sub_dim
        end = (i + 1) * sub_dim
        sub_vec = vector[start:end]
        
        # Find nearest centroid ID
        centroids = codebooks[i]
        dists = np.linalg.norm(centroids - sub_vec, axis=1)
        nearest_id = np.argmin(dists)
        encoded.append(np.uint8(nearest_id))
        
    return encoded
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
// Simplified conceptual code
func TrainPQ(vectors [][]float64, m int) [][][]float64 {
    // Splits vectors into m slices and runs k-means on each
    // Returns m codebooks, each with 256 centroids
    return trainKMeansPerSubspace(vectors, m) 
}

func EncodePQ(vector []float64, codebooks [][][]float64) []byte {
    m := len(codebooks)
    subDim := len(vector) / m
    encoded := make([]byte, m)
    
    for i := 0; i < m; i++ {
        subVec := vector[i*subDim : (i+1)*subDim]
        // Find closest centroid index (0-255)
        encoded[i] = findNearestCentroid(subVec, codebooks[i])
    }
    return encoded
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

using Vector = std::vector<double>;
using Codebook = std::vector<Vector>; // 256 centroids per subspace

// Simplified conceptual structure
std::vector<uint8_t> encode_pq(const Vector& vector, const std::vector<Codebook>& codebooks) {
    int m = codebooks.size();
    int sub_dim = vector.size() / m;
    std::vector<uint8_t> encoded;
    encoded.reserve(m);

    for (int i = 0; i < m; ++i) {
        // Extract sub-vector
        auto start = vector.begin() + (i * sub_dim);
        auto end = start + sub_dim;
        Vector sub_vec(start, end);

        // Find nearest centroid ID (0-255)
        double min_dist = std::numeric_limits<double>::max();
        uint8_t best_id = 0;

        for (int id = 0; id < 256; ++id) {
            double dist = 0.0;
            // Euclidean distance
            for (int k = 0; k < sub_dim; ++k) {
                double diff = sub_vec[k] - codebooks[i][id][k];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_id = static_cast<uint8_t>(id);
            }
        }
        encoded.push_back(best_id);
    }
    return encoded;
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

The snippets show the training and encoding steps instead of hiding them
behind a library call. They omit representative sampling, empty clusters,
rotations, parallel training, serialization, input validation, and optimized
kernels.

**Asymmetric distance computation (ADC)** leaves the query uncompressed. First
consider direct PQ, where codewords approximate the original database
subvectors. Let \(w_a^{(j)}\) mean codeword \(a\) in subspace \(j\). For each
query subvector, the system computes its distance to every codeword in that
subspace and stores those values in a lookup table. To score a database code,
it uses each stored byte to select one table entry and sums the selected
entries:

\[
\widehat d(q,x)^2
=\sum_{j=1}^{m}
\left\lVert q^{(j)}-w^{(j)}_{\operatorname{code}_j(x)}\right\rVert_2^2.
\]

Residual IVF-PQ needs one extra transformation. Suppose vector \(x\) belongs
to inverted list \(\ell\), whose coarse centroid is \(c_\ell\). Its PQ code
approximates the residual \(x-c_\ell\), not \(x\) itself. Therefore, while
scanning that list, build the lookup table from the **residual query**
\(q-c_\ell\):

\[
\widehat d(q,x)^2
=\sum_{j=1}^{m}
\left\lVert
(q-c_\ell)^{(j)}-w^{(j)}_{\operatorname{code}_j(x)}
\right\rVert_2^2,
\qquad x\text{ assigned to list }\ell.
\]

In either form, the table entries are exact distances from query subvectors to
codewords. However, their sum only **approximates** the distance from \(q\) to
the original \(x\). The uncompressed query prevents additional query-side
quantization error.

A common pipeline retrieves more than \(k\) candidates with PQ, then fetches original or higher-precision vectors and re-ranks them. The oversampling factor is a tuning parameter; \(10k\) is an example, not a law.

{{< mermaid >}}
graph TD
    subgraph Original ["Original Vector (384d)"]
        V["v1, v2, ... v8, v9 ... v16, ..."]
    end
    
    subgraph Split ["Split into Sub-spaces"]
        S1["Sub-vector 1 (8d)"]
        S2["Sub-vector 2 (8d)"]
        Sn["Sub-vector 48"]
    end
    
    subgraph Codebook ["Codebook Lookup (k-means)"]
        C1{"Find Nearest Centroid"}
        C2{"Find Nearest Centroid"}
        Cn{"Find Nearest Centroid"}
    end
    
    subgraph Compressed ["Compressed (IDs)"]
        ID1["ID: 42 (1 byte)"]
        ID2["ID: 15 (1 byte)"]
        IDn["ID: 99 (1 byte)"]
    end
    
    V --> S1
    V --> S2
    V -.-> Sn
    
    S1 --> C1 --> ID1
    S2 --> C2 --> ID2
    Sn --> Cn --> IDn
    
    style V fill:#f9f,stroke:#333
    style ID1 fill:#ccf,stroke:#333
    style ID2 fill:#ccf,stroke:#333
    style IDn fill:#ccf,stroke:#333
{{< /mermaid >}}

*Figure reading guide: one centroid ID replaces each eight-dimensional
subvector. The 48 IDs form the compressed code. The diagram omits codebook
storage and the optional coarse centroid used by IVF-PQ.*

#### The “FS” in IVF-PQFS: Fast Scan

An 8-bit PQ distance table looks small. However, scoring can require many
data-dependent table accesses. **Fast Scan** layouts reorganize codes and
lookup tables for single-instruction, multiple-data (SIMD) execution.

One common design uses 4-bit subquantizers:

* 4 bits select one of 16 codewords, so two codes fit in one byte. Four bits
  form a **nibble** or **nybble**.
* The 16-entry distance tables are small enough for register-oriented shuffle instructions on supported CPUs.
* The implementation stores codes in blocks so that it can score many
  candidates in parallel.

Fast Scan does not eliminate lookup. It replaces a cache-sensitive memory
lookup with an in-register shuffle or permute lookup. It also uses saturating
or quantized accumulators. The exact layout and supported instruction set
depend on the implementation.

A 4-bit subquantizer is coarser than an 8-bit subquantizer of the same
subvector width. The faster scan can examine more candidates or lists within
the latency budget. Measurements must show whether this improves the final
recall-throughput curve.

{{< mermaid >}}
graph TD
    subgraph RAM ["Packed Memory (1 Byte)"]
        Byte["Byte: 0xAB"]
        Hi["High Nibble: 10 (Code A)"]
        Lo["Low Nibble: 11 (Code B)"]
    end

    subgraph SIMD ["SIMD Register (Lookup Table Loaded)"]
        Reg["| Dist 0 | ... | Dist 10 (0.9) | Dist 11 (0.3) | ... | Dist 15 |"]
    end
    
    subgraph Compute ["Distance Calculation (In Register)"]
        D1["Dist for Code A: 0.9"]
        D2["Dist for Code B: 0.3"]
    end
    
    Byte --> Hi
    Byte --> Lo
    
    Hi -- "Shuffle/Permute" --> Reg --> D1
    Lo -- "Shuffle/Permute" --> Reg --> D2
    
    style Byte fill:#f9f,stroke:#333
    style Reg fill:#ccf,stroke:#333
    style D1 fill:#ff9,stroke:#333
    style D2 fill:#ff9,stroke:#333
{{< /mermaid >}}

*Figure reading guide: hexadecimal `0xAB` contains two 4-bit codes, `A = 10` and `B = 11`. SIMD shuffle instructions select the corresponding entries from a small distance table already loaded into a register.*

### Scalar quantization (SQ)

Scalar quantization maps each coordinate independently to a smaller numeric
representation, such as `int8`. Replacing `float32` with one byte per
coordinate gives a 4× code-only reduction. PQ replaces a block with one
cluster ID. In contrast, SQ gives each original coordinate one quantized
value.

A dataset-wide calibration can use one scale for all values. It can also use a
separate scale for each dimension. The best design depends on the distribution
and distance kernel. The snippets use a **per-vector** minimum and maximum to
show the affine mapping.

For a nonconstant vector, let
\(x_{\min}=\min_i x_i\), \(x_{\max}=\max_i x_i\), and
\(s=255/(x_{\max}-x_{\min})\). Define the stored integer code \(z_i\) as

\[
z_i=
\operatorname{clip}_{[-128,127]}
\left(
\left\lfloor (x_i-x_{\min})s+\frac{1}{2}\right\rfloor-128
\right).
\]

The scaled value before rounding is nonnegative, so the floor expression
specifies **round half up** consistently in all three languages below. The
encoding rule is deterministic; information is lost when the integer code is
reconstructed as a floating-point value.

Each vector in this example gets a separate scale and offset. Therefore, the
integer codes do not share one coordinate system. A distance routine must
dequantize the codes or use each vector's calibration parameters. This
teaching version is not suitable for direct use in a database index.

{{< mermaid >}}
graph LR
    subgraph F ["Float32 Vector"]
        F1["0.12"]
        F2["0.98"]
        F3["-0.5"]
    end
    
    subgraph Map ["Map Range [min, max]"]
        M1["clip(floor((x - min) * scale + 0.5) - 128)"]
    end
    
    subgraph I ["Int8 Vector"]
        I1["-21"]
        I2["127"]
        I3["-128"]
    end
    
    F1 --> M1 --> I1
    F2 --> M1 --> I2
    F3 --> M1 --> I3
    
    style F1 fill:#f9f,stroke:#333
    style I1 fill:#ccf,stroke:#333
{{< /mermaid >}}

*Figure reading guide: the example calibrates one vector whose minimum is
\(-0.5\) and maximum is \(0.98\). Round-half-up maps \(0.12\) to \(-21\),
\(0.98\) to 127, and \(-0.5\) to -128. The equation above also defines
clipping and the zero point; a concrete implementation must separately define
the constant-vector case.*

For a nonconstant vector, approximate reconstruction is

\[
\widehat x_i=\frac{z_i+128}{s}+x_{\min}.
\]

The constant-vector case needs a separate rule because its scale is zero.

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import math

def scalar_quantize(vector):
    # Map floats to int8 (-128 to 127) using per-vector min/max
    min_val, max_val = min(vector), max(vector)
    scale = 0 if max_val == min_val else 255 / (max_val - min_val)
    
    quantized = []
    for x in vector:
        code = math.floor((x - min_val) * scale + 0.5) - 128
        quantized.append(max(-128, min(127, code)))
        
    return quantized, scale, min_val
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
import "math"

func ScalarQuantize(vector []float64) ([]int8, float64, float64) {
	minVal, maxVal := vector[0], vector[0]
	for _, v := range vector {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	var scale float64
	if maxVal > minVal {
		scale = 255.0 / (maxVal - minVal)
	}
	quantized := make([]int8, len(vector))

	for i, v := range vector {
		q := int(math.Round((v-minVal)*scale)) - 128
		if q > 127 { q = 127 }
		if q < -128 { q = -128 }
		quantized[i] = int8(q)
	}

	return quantized, scale, minVal
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

struct QuantizedResult {
    std::vector<int8_t> quantized;
    double scale;
    double min_val;
};

QuantizedResult scalar_quantize(const std::vector<double>& vector) {
    double min_val = vector[0];
    double max_val = vector[0];
    for (double v : vector) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    double scale = (max_val > min_val) ? 255.0 / (max_val - min_val) : 0.0;
    std::vector<int8_t> quantized;
    quantized.reserve(vector.size());

    for (double v : vector) {
        int q = static_cast<int>(std::lround((v - min_val) * scale)) - 128;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        quantized.push_back(static_cast<int8_t>(q));
    }

    return {quantized, scale, min_val};
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

### Binary quantization (BQ)

A sign-based binary quantizer records one bit per coordinate:

\[
b_i(x)=
\begin{cases}
1,&x_i>0,\\
0,&x_i\le0.
\end{cases}
\]

Compared with 32-bit floats, packed bits give a 32× code-only reduction.
Compare two binary codes with XOR followed by a **population count**. This
operation counts the bit positions that differ. Modern CPUs can perform this
Hamming-distance operation efficiently.

This simple rule preserves signs, not magnitudes. Some systems first center,
rotate, project, or learn the binary representation. These transformations can
prevent informative directions from collapsing onto an unsuitable threshold.
They are design choices, not universal requirements.

{{< mermaid >}}
graph TD
    subgraph Input ["Float32 Input (4D Example)"]
        V1["0.8"]
        V2["-0.2"]
        V3["0.1"]
        V4["-0.9"]
    end
    
    subgraph Sign ["Sign Bit Threshold (> 0)"]
        S1{> 0?}
        S2{> 0?}
        S3{> 0?}
        S4{> 0?}
    end
    
    subgraph Output ["Binary Output (1 bit per dim)"]
        B1["1"]
        B2["0"]
        B3["1"]
        B4["0"]
    end
    
    V1 --> S1 --> B1
    V2 --> S2 --> B2
    V3 --> S3 --> B3
    V4 --> S4 --> B4
    
    style B1 fill:#ccf,stroke:#333
    style B2 fill:#fcc,stroke:#333
    style B3 fill:#ccf,stroke:#333
    style B4 fill:#fcc,stroke:#333
{{< /mermaid >}}

*Figure reading guide: the sign threshold converts four floats into the bit pattern `1010`. A packed implementation stores many such bits in machine words rather than one byte per Boolean as the Python teaching code does.*

{{< code-tabs >}}
{{< code-tab "Python" >}}
```python
import numpy as np

def binary_quantize(vector):
    # If x > 0 return 1, else 0
    return np.where(vector > 0, 1, 0).astype(np.int8)

def hamming_distance(bq1, bq2):
    # XOR and count bits (popcount)
    # In pure Python/Numpy this is naive; 
    # production systems use packed bits + CPU instructions
    xor_diff = np.bitwise_xor(bq1, bq2)
    return np.sum(xor_diff)
```
{{< /code-tab >}}
{{< code-tab "Go" >}}
```go
import "math/bits"

func BinaryQuantize64(vector []float64) uint64 {
    // Teaching helper: pack only the first 64 dimensions.
    var mask uint64 = 0
    n := len(vector)
    if n > 64 {
        n = 64
    }
    for i := 0; i < n; i++ {
        if vector[i] > 0 {
            mask |= (uint64(1) << uint(i))
        }
    }
    return mask
}

func HammingDistance(a, b uint64) int {
    // XOR + PopCount (Hardware accelerated)
    return bits.OnesCount64(a ^ b)
}
```
{{< /code-tab >}}
{{< code-tab "C++" >}}
```cpp
#include <vector>
#include <cstdint>

// Teaching helper: pack only the first 64 dimensions.
uint64_t binary_quantize(const std::vector<double>& vector) {
    uint64_t mask = 0;
    for (size_t i = 0; i < vector.size() && i < 64; ++i) {
        if (vector[i] > 0) {
            mask |= (1ULL << i);
        }
    }
    return mask;
}

int hamming_distance(uint64_t a, uint64_t b) {
    // XOR + PopCount (CPU instruction)
    // GCC/Clang intrinsic for efficient population count
    return __builtin_popcountll(a ^ b);
    // In C++20, you can use: std::popcount(a ^ b);
}
```
{{< /code-tab >}}
{{< /code-tabs >}}

The Go and C++ examples stop after 64 dimensions to show one machine word. A
production encoder emits an array of words. It must also handle a partial final
word. Binary quantization can lose substantial recall, even at high dimension.
High dimension does not guarantee that sign bits preserve the desired ranking.
Systems commonly combine binary quantization with oversampling and
higher-precision re-ranking.

### Anisotropic vector quantization and ScaNN

Google's **ScaNN** system combines partitioning, quantized scoring, and
reordering. Its anisotropic vector quantization targets errors that can disturb
maximum-inner-product rankings. This goal is more specific than minimizing
ordinary reconstruction error <a id="cite-scann-icml"></a>[[scann-icml]](#ref-scann-icml).

Let \(\tilde x\) be a quantized approximation to database vector \(x\), with error \(e=x-\tilde x\). For any query \(q\), the inner-product error is

\[
q^\top x-q^\top\tilde x=q^\top e.
\]

Decompose \(e\) into a component parallel to \(x\) and a component orthogonal to \(x\):

\[
e=e_{\parallel}+e_{\perp}.
\]

For queries likely to produce high inner products with \(x\), the parallel component tends to matter more to ranking. The anisotropic objective therefore penalizes parallel error more strongly than orthogonal error under the paper's query model.

{{< mermaid >}}
graph TD
    subgraph Conceptual ["Loss Minimization Strategy"]
        X(Data Vector X)
        Err_Para["Parallel error (higher weight)"]
        Err_Ortho["Orthogonal error (lower weight, not zero)"]
        
        X -- "High Weight (Critical)" --> Err_Para
        X -- "Lower Weight" --> Err_Ortho
        
        Err_Para --> Bad["Larger expected score disruption under the query model"]
        Err_Ortho --> OK["Smaller expected score disruption under the query model"]
    end
    
    style Bad fill:#fcc,stroke:#333
    style OK fill:#ccf,stroke:#333
    style Err_Para fill:#ff9,stroke:#333
    style Err_Ortho fill:#eef,stroke:#333
{{< /mermaid >}}

*Figure reading guide: the diagram decomposes quantization error relative to
the database vector \(x\). “Low weight” does not mean that orthogonal error is
harmless. The loss weights each direction according to its expected ranking
effect under the model.*

A simplified schematic loss is

\[
\mathcal L
\approx w_{\parallel}\lVert e_{\parallel}\rVert_2^2
+w_{\perp}\lVert e_{\perp}\rVert_2^2,
\qquad w_{\parallel}>w_{\perp}.
\]

The paper derives the weights from a probabilistic query model. It does not
select one universal pair of weights. The objective assigns more of the finite
compression budget to directions that can change the score. Achieved recall
still depends on training data, partitioning, code size, candidate budget, and
reordering.

---

## Reproducibility: separate the sources of change

“ANN is nondeterministic” is too blunt to be useful. There are at least three separate questions:

1. **Can two builds differ?** Yes. Random levels, randomized clustering, insertion order, parallel construction, and tie handling can change the stored index.
2. **Can two queries against one immutable index differ?** Often they do not
   when the implementation, hardware, thread schedule, and tie-breaking are
   fixed. They can differ when parallel work is nondeterministic or tie
   ordering is unstable.
3. **Can the live corpus change between queries?** Yes. Updates, deletes, segment merges, and snapshot boundaries can change both the candidate graph and the exact ground truth.

### Build order and live mutation

HNSW topology depends on insertion order. Each new node links into the existing
graph. IVF and PQ training can depend on random initialization and sample
order. Some reproducible builds require a fixed seed. A fixed seed is not
always sufficient. Parallel races or different floating-point reduction orders
can still change a build.

A live update creates a different problem: the two searches may not observe the same logical index version.

    {{< mermaid >}}
    graph TD
        classDef hit fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000;
        classDef writer fill:#f3e5f5,stroke:#ab47bc,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
        
        subgraph "Timing A: Search is Faster (Returns Node B)"
            direction TB
            S1[Search Thread] -->|1. Routes to| A1((Node A))
            A1 -->|2. Routes to| B1((Node B))
            W1[Writer Thread] -.->|3. Inserts| C1((Node C))
            class B1 hit;
            class W1,C1 writer;
        end

        subgraph "Timing B: Writer is Faster (Returns Node C)"
            direction TB
            S2[Search Thread] -->|2. Routes to| A2((Node A))
            W2[Writer Thread] -->|1. Inserts & Wires| C2((Node C))
            A2 -.->|New Edge Created| C2
            A2 -->|3. Finds better path| C2
            class C2 hit;
            class W2 writer;
        end
    {{< /mermaid >}}

*Figure reading guide: both timelines are internally valid, but they expose different index states. Timing A completes before Node C is visible; Timing B routes after the new edge exists. Snapshot isolation can make the visibility rule explicit.*

### Floating-point order

IEEE 754 floating-point addition is not associative in general:

\[
(a+b)+c\ne a+(b+c).
\]

SIMD kernels often accumulate partial sums in lanes and then reduce those lane totals as a tree. A scalar kernel may add left to right. Compilers can also contract multiply-add operations, reassociate expressions under fast-math settings, or select different vector widths. Usually the numerical difference is tiny. It matters when two candidate scores are close enough that the rounding difference crosses their sorting boundary.
    
The following C++ example is intentionally extreme so the rounding-order effect is visible. The exact real-number sum is 16,777,220. The scalar order repeatedly loses each `1.0f`; the tree order first combines the four small terms into an exactly representable `4.0f`.

The example uses Intel Streaming SIMD Extensions (SSE) intrinsics. Advanced
Vector Extensions (AVX) provide wider related instruction families on
supporting processors.

The standalone source is available as [`simd_demo.cpp`](simd_demo.cpp).

```cpp
#include <immintrin.h> // Intel SSE/AVX hardware intrinsics

// 16777216.0 is 2^24. The next explicitly representable float32 is 16777218.0.
// At this threshold, float32 stops being able to represent odd numbers.
alignas(16) float values[5] = {16777216.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// 1. Sequential Loop (Standard Scalar Addition)
// Under the default round-to-nearest, ties-to-even mode,
// 16777216.0f + 1.0f is halfway between representable values
// and rounds back to 16777216.0f.
float scalar_sum = 0.0f;
for(int i=0; i < 5; i++) scalar_sum += values[i];
// Result: 16777216.0 (Each addition rounds back to the same value.)

// 2. Native Hardware SIMD Addition (SSE Horizontal Reduction)
// Load the four trailing 1.0f values into a 128-bit SIMD register.
__m128 sum_vec = _mm_loadu_ps(&values[1]);
sum_vec = _mm_hadd_ps(sum_vec, sum_vec); // Horizontal sum: [A+B, C+D, A+B, C+D]
sum_vec = _mm_hadd_ps(sum_vec, sum_vec); // Final reduction: [A+B+C+D, ...]

// The CPU accumulates the four 1.0f values into 4.0f before
// adding the much larger value.
float tree_sum = values[0] + _mm_cvtss_f32(sum_vec);
// Result: 16777220.0 (this reduction order preserved the +4.0)
```

### Define the reproducibility contract

For a repeatable experiment, record:

* corpus snapshot and metadata-filter snapshot;
* vector IDs, insertion order, training sample, and random seeds;
* embedding model and preprocessing version;
* metric, normalization, tie policy, and \(k\);
* index implementation, version, build parameters, and search parameters;
* central processing unit (CPU) or graphics processing unit (GPU) model,
  instruction path, compiler flags, and thread counts;
* cache state and whether concurrent mutation was allowed.

An immutable snapshot, fixed build inputs, stable tie-breaking, and pinned
numerical kernels can make repeated results far more stable. “Deterministic
Basic Linear Algebra Subprograms (BLAS)” is not a magic switch that guarantees
every index layer is deterministic.

Exact re-ranking makes the **ordering inside a bounded ANN candidate set**
exact. It cannot recover a true neighbor that the ANN stage did not propose.
An exact top-\(k\) guarantee requires broader coverage. The exact stage must
cover the full eligible corpus or a partition that provably contains the
answer.

---

## Measure and tune ANN with an independent baseline

An ANN index is not “fast” or “accurate” by itself. Its result is one point on
a measured tradeoff curve. The curve depends on the dataset, workload, and
machine.

### 1. Fix the exact comparison

Create query sets that are separate from index-training data. Use one **tuning set** while choosing parameters, then report final numbers on an untouched **test set**.

For every test query, run exact top-\(k\) search over the same:

* corpus snapshot;
* metadata filter;
* vector preprocessing and normalization;
* distance or similarity function;
* tie-breaking rule.

The ANN query and the exact baseline must use the same filters. Otherwise, the
resulting Recall@\(k\) value is invalid.

### 2. Measure candidate recall and user relevance separately

The ANN accuracy contract defines

\[
\operatorname{Recall@}k(q)=\frac{|A_k(q)\cap T_k(q)|}{k}.
\]

Average it over the query set. Also inspect percentiles or the distribution: a mean of 0.95 can hide a minority of catastrophic zero-recall queries.

Recall against exact vector neighbors is a systems metric. It does not
establish that either result set is relevant to a human. Evaluate downstream
relevance separately. Use judgments or task metrics such as normalized
discounted cumulative gain (nDCG) and mean reciprocal rank (MRR). Other
measures include answer support, conversion, or an application-specific
metric.

### 3. Sweep one search curve, then revisit build choices

For a fixed built index:

* sweep HNSW `ef_search`;
* sweep IVF `nprobe`;
* sweep DiskANN beam/search-width and cache choices;
* sweep oversampling and re-ranking depth for quantized systems.

Then compare several build configurations such as HNSW \(M\) and `ef_construction`, IVF `nlist`, PQ code size, and Vamana degree/pruning settings. A wider query search cannot always repair a poorly built index.

### 4. Measure the whole operating envelope

Record at least:

* median, p95, and p99 latency;
* throughput at controlled concurrency;
* CPU, RAM, SSD space, and SSD read volume;
* build time and peak build memory;
* update, delete, and compaction costs;
* cold-cache and warm-cache behavior;
* performance with realistic filters and batch sizes.

Warm up deliberately, repeat runs, randomize or pair configuration order, and
report variability. For randomized graph builds, repeat across seeds or
insertion orders. This prevents one unusually favorable topology from
representing the complete algorithm.

Plotting Recall@\(k\) against queries per second (QPS) or tail latency gives a
Pareto curve. A point is dominated if another configuration has at least as
much recall and better performance. Choose from the non-dominated frontier
according to the product's recall target and resource budget.

---

## Compare the families

Universal millisecond claims hide the variables that matter, so this table compares mechanisms rather than promising latency.

| Family | Candidate-reduction mechanism | Dominant storage | Main query knob | Characteristic failure mode |
|:--|:--|:--|:--|:--|
| Exact flat search | None; score every eligible vector | Full vectors | Batch size and hardware kernel | Cost grows with eligible \(N\times d\) |
| HNSW | Route through a layered proximity graph | Full or quantized vectors plus graph | `ef_search` | Local routing misses; graph/filter fragmentation |
| Vamana / DiskANN | Route through a degree-bounded flat graph designed with SSD I/O | SSD graph/full vectors plus DRAM codes/cache | Beam/search width and cache budget | SSD stalls or graph-routing misses |
| IVF-Flat | Probe selected centroid lists | Full vectors plus centroids/list IDs | `nprobe` | True neighbor lies in an unprobed cell |
| IVF-PQ | Probe lists and score compressed codes | PQ codes, centroids, optional full vectors | `nprobe`, code bytes, rerank depth | Boundary miss plus quantization misranking |
| IVF-PQ Fast Scan | IVF-PQ with SIMD-oriented code/table layouts | Usually compact 4-bit PQ blocks | `nprobe` and scan budget | Coarser codes or unsupported hardware path |
| ScaNN-style partition/quantize/reorder | Partition, anisotropic quantized scoring, then optional exact reorder | Codes plus reorder payload | leaves/candidates and reorder count | Partition or quantization stage drops a needed candidate |

### Estimate HNSW memory instead of quoting a percentage

For raw vectors,

\[
\text{vector bytes}=N\,d\,b,
\]

where \(b\) is bytes per coordinate. A rough adjacency estimate is

\[
\text{link bytes}
\approx
N\left(M_0+\frac{M}{M-1}\right)s,
\]

This estimate assumes that Layer 0 stores about \(M_0\) links. It also assumes
an \(M^{-\ell}\) population tail for upper layers. Each participating
upper-layer node stores about \(M\) links. Each neighbor identifier costs \(s\)
bytes. The upper-layer series is

\[
M\sum_{\ell=1}^{\infty}M^{-\ell}=\frac{M}{M-1}.
\]

This is an explanatory estimate, not an allocation formula. Implementations
add list lengths, levels, alignment, locks, visited-state structures, labels,
deleted-node state, and allocator overhead. They can also use 4-byte IDs
instead of 8-byte pointers.

For \(N=10^6\), \(d=768\), `float32`, \(M=16\), \(M_0=32\), and an illustrative \(s=8\):

\[
\text{vectors}=10^6\times768\times4=3.072\text{ GB},
\]

\[
\text{links}\approx10^6\left(32+\frac{16}{15}\right)8
\approx264.5\text{ MB}.
\]

The graph is about 8.6% of the raw vector bytes in that narrow example before
overhead. If compression reduces the vector payload to 128 `int8` bytes, the
link size does not necessarily change. Graph bytes can then exceed vector
bytes. Therefore, “HNSW adds 40%” is not a portable statement.

## Algorithm selection and production realities {#algorithm-selection--production-realities}

The mathematics narrows the field. The workload makes the final choice.

### Filters can change the search problem

A product query is often “nearest vectors among documents visible to tenant 42 from the last 30 days,” not “nearest vectors in the whole corpus.”

There are several filter execution strategies:

* **Pre-filter:** determine eligible IDs first, then run exact or ANN search only over them. This is excellent when the eligible set is small, but an ANN graph restricted to those IDs may become disconnected.
* **Post-filter:** run ANN over the whole index, then discard ineligible results. This preserves graph navigation but may return fewer than \(k\) results or low recall when selectivity is high.
* **Filter-aware traversal:** carry a bitmap or predicate through search and expand more candidates as needed.
* **Partitioned or filtered indexes:** build separate physical structures for common tenants or predicates.

No strategy wins for every selectivity. Spanner documents filter columns
stored with its vector index. This design permits filtering at the leaf level
<a id="cite-spanner-vector-indexes"></a>[[spanner-vector-indexes]](#ref-spanner-vector-indexes).
pgvector documents iterative scans. These scans continue an approximate search
when post-filtering removes too many rows <a id="cite-pgvector-github"></a>[[pgvector-github]](#ref-pgvector-github).

### Hybrid search solves a different weakness

Dense embeddings are useful for semantic paraphrase. Lexical retrieval is
useful for exact product IDs, names, quoted phrases, and rare tokens.
**Hybrid search** runs dense and lexical retrieval and then combines the
rankings.

Okapi BM25 is a lexical ranking function. It rewards a document when the
document contains query terms, and it gives more weight to terms that are rare
in the collection. Repeated matches increase a term's contribution, but the
increase eventually saturates. BM25 also normalizes for document length, so a
long document does not win only because it contains more words.

One common BM25 form for short keyword queries is

\[
\operatorname{BM25}(D,Q)
=\sum_{t\in Q}
\operatorname{IDF}(t)
\frac{f(t,D)(k_1+1)}
{f(t,D)+k_1\left(1-b+b\frac{|D|}{\operatorname{avgdl}}\right)}.
\]

Here, \(D\) is one candidate document, \(Q\) is the query, and \(t\) is one
query term. The value \(f(t,D)\) counts occurrences of \(t\) in \(D\).
The document length is \(|D|\), and \(\operatorname{avgdl}\) is the average
document length in the indexed collection. The inverse document frequency
\(\operatorname{IDF}(t)\) gives more influence to a term that occurs in fewer
documents. Implementations use several closely related IDF conventions.

The positive parameter \(k_1\) controls how quickly repeated matches saturate.
The parameter \(b\), usually constrained to \(0\le b\le1\), controls document
length normalization. Setting \(b=0\) disables that normalization in this
formula. Setting \(b=1\) applies the full document-length ratio
\(|D|/\operatorname{avgdl}\). These term-frequency and length-normalization
ideas are central to BM25 <a id="cite-bm25-framework"></a>[[bm25-framework]](#ref-bm25-framework).

BM25 does not directly recognize a paraphrase when the query and document
share no useful terms. Dense retrieval can connect such inputs when the
embedding model places them near each other. Dense retrieval can still miss an
exact identifier or a rare literal token. The two methods therefore fail in
different ways. This is why hybrid retrieval can help.

One score-free fusion method is Reciprocal Rank Fusion:

\[
\operatorname{RRF}(d)
=\sum_{r\in\mathcal R:\,d\in r}
\frac{1}{c+\operatorname{rank}_r(d)}.
\]

Here, \(\mathcal R\) is the set of ranked lists.
\(\operatorname{rank}_r(d)\) is the one-based rank of document \(d\) in list
\(r\). A document that is absent from a list contributes zero because that
list is not included in the sum for the document. The positive constant \(c\)
reduces the advantage of the first few positions. RRF does not assume that
BM25 and cosine scores share a calibrated scale
<a id="cite-rrf-paper"></a>[[rrf-paper]](#ref-rrf-paper). RRF is a strong
baseline, not an automatic optimum. Learned fusion and re-rankers can perform
better when good judgments are available.

### Updates and deletes have physical costs

An index is a maintained data structure, not a one-time benchmark artifact.

* HNSW insertion changes local topology and can make quality depend on arrival order.
* Deletion may use tombstones until background repair or rebuilding reclaims the node and its edges.
* IVF centroids can become stale when the distribution drifts; list sizes can become imbalanced.
* PQ codebooks can become mismatched to new data.
* Segment-oriented systems may search old and new segments together, then merge them later.
* SSD indexes must balance foreground reads against compaction, rebuilding, and write amplification.

Measure freshness delay, update throughput, delete visibility, rebuild duration,
and degraded recall during maintenance, not only steady-state reads.

### A decision path based on measured constraints

Fast SSD indexes often use the Non-Volatile Memory Express (NVMe) storage
interface. Its random-read behavior is part of a DiskANN benchmark. It is not
a background implementation detail.

{{< mermaid >}}
graph TD
    A["Start with exact filtered search"] --> B{"Meets latency and cost target?"}
    B -->|"Yes"| C["Keep exact search and measure growth"]
    B -->|"No"| D{"Graph plus vectors fit the RAM budget?"}
    D -->|"Yes"| E["Benchmark HNSW against IVF / ScaNN-style partitioning"]
    D -->|"No"| F{"NVMe random-read budget acceptable?"}
    F -->|"Yes"| G["Benchmark DiskANN / Vamana"]
    F -->|"No"| H["Use stronger compression and partition pruning"]
    E --> I{"Heavy or selective metadata filters?"}
    G --> I
    H --> I
    I -->|"Yes"| J["Benchmark pre-, post-, and filter-aware execution"]
    I -->|"No"| K["Sweep recall–latency frontier"]
    J --> K
    K --> L["Validate updates, cold cache, and tail latency"]
{{< /mermaid >}}

*Figure reading guide: row count is deliberately absent. Begin with a measured
exact baseline. Then use observed cost, memory, storage, filters, and
maintenance to select a branch. One million vectors can be easy or difficult
to search. The result depends on dimension, selectivity, hardware, and traffic.*

### Documented implementations in current products

The rows below describe algorithms and options explicitly named in first-party
documentation checked on the stated verification date. They establish
**documented capability**, not independent performance, default behavior in
every region, or continued availability in later releases. Preview status,
index combinations, configuration names, and automatic-selection behavior
should be rechecked against the exact product version before deployment. An
omission does not prove that a service lacks an internal technique.

| Product | Explicitly documented vector-search paths | Practical note |
|:--|:--|:--|
| Google Vector Search | ScaNN family | Google's overview connects the service to ScaNN; recheck the current product docs for the deployed region and API surface <a id="cite-vertex-ai-vector-search"></a>[[vertex-ai-vector-search]](#ref-vertex-ai-vector-search). |
| Google AlloyDB | ScaNN index; pgvector HNSW and IVFFlat paths | ScaNN exposes tree levels, leaves, and quantizer choices; IVFFlat exposes list tuning <a id="cite-alloydb-ivfflat"></a>[[alloydb-ivfflat]](#ref-alloydb-ivfflat) <a id="cite-alloydb-scann"></a>[[alloydb-scann]](#ref-alloydb-scann). |
| Google Spanner | Exact k-nearest-neighbor (kNN); tree-based ScaNN ANN vector indexes | Docs cover two- and three-level trees, leaf probing, PostgreSQL and Google Structured Query Language (GoogleSQL) syntax, and index-side filtering <a id="cite-spanner-knn"></a>[[spanner-knn]](#ref-spanner-knn) <a id="cite-spanner-ann"></a>[[spanner-ann]](#ref-spanner-ann). |
| Azure AI Search | HNSW; `exhaustiveKnn`; scalar and binary compression options | Profiles bind an algorithm and optional compression configuration; names and defaults are version-sensitive <a id="cite-azure-ai-search-vector-index"></a>[[azure-ai-search-vector-index]](#ref-azure-ai-search-vector-index). |
| Azure Cosmos DB for NoSQL | `flat`; `quantizedFlat`; `diskANN` | Small `quantizedFlat` or `diskANN` collections may fall back to full scan until the quantizer has enough vectors <a id="cite-cosmos-db-vector"></a>[[cosmos-db-vector]](#ref-cosmos-db-vector). |
| SQL Server / Azure SQL family | DiskANN vector index; exact vector-distance expressions | ANN index and `VECTOR_SEARCH` availability is version- and preview-dependent; check the product/version banner before relying on GA behavior <a id="cite-sql-server-vector"></a>[[sql-server-vector]](#ref-sql-server-vector). |
| Elasticsearch | Lucene HNSW dense-vector indexing; exact/scripted paths and quantized index options vary by mapping/version | Candidate count, filters, oversampling, and rescoring are exposed in the kNN workflow <a id="cite-elasticsearch-knn"></a>[[elasticsearch-knn]](#ref-elasticsearch-knn). |
| Databricks AI Search | HNSW ANN; exact-keyword BM25; RRF hybrid fusion | Documentation names HNSW and L2, with normalization required for cosine-equivalent ranking <a id="cite-databricks-mosaic-ai"></a>[[databricks-mosaic-ai]](#ref-databricks-mosaic-ai). |
| Milvus | FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW and quantized HNSW variants, SCANN, DiskANN | The documented menu is broad; availability depends on Milvus version and deployment mode <a id="cite-milvus-in-memory"></a>[[milvus-in-memory]](#ref-milvus-in-memory) <a id="cite-milvus-disk-index"></a>[[milvus-disk-index]](#ref-milvus-disk-index). |
| Weaviate | `hnsw`, `flat`, `dynamic`, and `hfresh` | `dynamic` starts flat and converts to HNSW; `hfresh` is a cluster-based design with an HNSW centroid index <a id="cite-weaviate-vector-index"></a>[[weaviate-vector-index]](#ref-weaviate-vector-index). |
| SingleStore | FLAT, IVF_FLAT, IVF_PQ, IVF_PQFS, HNSW_FLAT, HNSW_PQ | `AUTO` behavior is explicitly documented as subject to future change <a id="cite-singlestore-vector-indexing"></a>[[singlestore-vector-indexing]](#ref-singlestore-vector-indexing). |
| LanceDB | IVF_PQ, IVF_RQ, IVF_HNSW_FLAT/PQ/SQ, binary IVF_FLAT, and exhaustive bypass | Docs describe HNSW as a sub-index inside IVF partitions rather than a top-level index <a id="cite-lancedb-vector-indexes"></a>[[lancedb-vector-indexes]](#ref-lancedb-vector-indexes). |
| pgvector | Exact scan, HNSW, IVFFlat, half-precision and binary-quantized expression-index patterns | Filtering is applied after approximate scanning in common plans; iterative scans can continue when needed. Behavior depends on the installed pgvector release [[pgvector-github]](#ref-pgvector-github). |

**Verification date: 2026-07-27.** This is a dated documentation snapshot, not a
comparative benchmark. Products that do not publicly commit to an internal
algorithm are intentionally not assigned one here. “Not documented” and “not
supported” are different claims.

{{< panel "info" >}}
**Libraries, databases, and managed services are different comparison units**

Faiss is a library, not a managed database. It implements flat, IVF, PQ, Fast
Scan, HNSW, and GPU designs. A managed service also includes ingestion,
replication, filters, isolation, compaction, observability, and autoscaling.
Compare complete systems when selecting a product. Compare algorithm
implementations when studying the underlying tradeoffs.
{{< /panel >}}

## Further reading and benchmarks
Production results depend on hardware architecture, compiler optimizations,
and dataset distribution. Theory alone cannot select a production index.

* **[ANN-Benchmarks](https://ann-benchmarks.com/index.html)** compares
  approximate nearest-neighbor implementations across recall and queries per
  second. Its [GitHub repository](https://github.com/erikbern/ann-benchmarks)
  contains the benchmark source.
* **[VectorDBBench](https://github.com/zilliztech/VectorDBBench)** compares
  managed and open-source vector databases.

## Other high-performing ANN systems in leaderboards

[ANN-Benchmarks](https://ann-benchmarks.com/index.html) and similar
leaderboards compare *end-to-end implementations*, not only abstract
algorithms. Strong curves can result from:

* CPU/SIMD specialization, for example AVX-512;
* cache-aware adjacency layouts;
* aggressive quantization implementations;
* dataset-specific build and pruning heuristics;
* different search budgets, such as beam widths and candidate limits.

The main mathematical families are graph routing, partitioning, and
compression. HNSW and Vamana use graph routing. IVF uses partitioning. PQ, SQ,
and BQ use compression.

Two other leaderboard entries illustrate this design space:

* **Descartes (01.AI):** An in-memory ANN engine that combines a navigable graph
  with quantization and modern CPU instructions.
* **QSGNGT:** A graph-based approach that uses NGT-style graph search and an
  approximate k-nearest-neighbor graph (AKNNG) pipeline: AKNNG &rarr; search
  graph &rarr; quantized search graph.

These systems optimize different parts of the graph-ANN and compression design
space. Their mathematics shares features with HNSW, DiskANN, and graph
pruning. Systems engineering can strongly affect their leaderboard
differences.

## References

- **[malkov-hnsw]** **<a id="ref-malkov-hnsw"></a>Malkov, Y. A., & Yashunin, D. A. (2018).** *[Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320).* Institute of Electrical and Electronics Engineers Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI). [↩](#cite-malkov-hnsw)
- **[diskann-neurips]** **<a id="ref-diskann-neurips"></a>Subramanya, S. J., et al. (2019).** *[DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/).* Conference on Neural Information Processing Systems (NeurIPS). [↩](#cite-diskann-neurips)
- **[scann-icml]** **<a id="ref-scann-icml"></a>Guo, R., et al. (2020).** *[Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)](https://arxiv.org/abs/1908.10396).* International Conference on Machine Learning (ICML). [↩](#cite-scann-icml)
- **[faiss-paper]** **<a id="ref-faiss-paper"></a>Johnson, J., et al. (2019).** *[Billion-scale similarity search with GPUs (Faiss)](https://arxiv.org/abs/1702.08734).* IEEE Transactions on Big Data. [↩](#cite-faiss-paper)
- **[aggarwal-surprising]** **<a id="ref-aggarwal-surprising"></a>Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001).** *[On the Surprising Behavior of Distance Metrics in High Dimensional Space](https://bib.dbvis.de/uploadedFiles/155.pdf).* International Conference on Database Theory (ICDT). [↩](#cite-aggarwal-surprising)
- **[vertex-ai-vector-search]** **<a id="ref-vertex-ai-vector-search"></a>Google Cloud.** *[Vector Search overview | Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview).* [↩](#cite-vertex-ai-vector-search)
- **[sql-server-vector]** **<a id="ref-sql-server-vector"></a>Microsoft.** *[Vector Search & Vector Index - SQL Server](https://learn.microsoft.com/en-us/sql/sql-server/ai/vectors?view=sql-server-ver17).* [↩](#cite-sql-server-vector)
- **[azure-ai-search-store]** **<a id="ref-azure-ai-search-store"></a>Microsoft.** *[Vector indexes in Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-store).*
- **[milvus-disk-index]** **<a id="ref-milvus-disk-index"></a>Milvus.** *[DISKANN | Milvus Documentation](https://milvus.io/docs/diskann.md).* [↩](#cite-milvus-disk-index)
- **[singlestore-indexed-ann]** **<a id="ref-singlestore-indexed-ann"></a>SingleStore.** *[Announcing SingleStore Indexed ANN Vector Search](https://www.singlestore.com/blog/singlestore-indexed-ann-vector-search/).* [↩](#cite-singlestore-indexed-ann)
- **[singlestore-tuning]** **<a id="ref-singlestore-tuning"></a>SingleStore.** *[Tuning Vector Indexes and Queries](https://docs.singlestore.com/cloud/developer-resources/functional-extensions/tuning-vector-indexes-and-queries/).* [↩](#cite-singlestore-tuning)
- **[alloydb-ivfflat]** **<a id="ref-alloydb-ivfflat"></a>Google Cloud.** *[Create an IVFFlat index | AlloyDB for PostgreSQL](https://docs.cloud.google.com/alloydb/docs/ai/create-ivfflat-index).* [↩](#cite-alloydb-ivfflat)
- **[alloydb-scann]** **<a id="ref-alloydb-scann"></a>Google Cloud.** *[AlloyDB ScaNN Index reference](https://docs.cloud.google.com/alloydb/docs/reference/ai/scann-index-reference).* [↩](#cite-alloydb-scann)
- **[spanner-vector-indexes]** **<a id="ref-spanner-vector-indexes"></a>Google Cloud.** *[Create and manage vector indexes | Spanner](https://docs.cloud.google.com/spanner/docs/vector-indexes).* [↩](#cite-spanner-vector-indexes)
- **[spanner-knn]** **<a id="ref-spanner-knn"></a>Google Cloud.** *[Perform vector similarity search in Spanner by finding the K-nearest neighbors](https://docs.cloud.google.com/spanner/docs/find-k-nearest-neighbors).* [↩](#cite-spanner-knn)
- **[spanner-ann]** **<a id="ref-spanner-ann"></a>Google Cloud.** *[Find approximate nearest neighbors (ANN) and query vector embeddings | Spanner](https://docs.cloud.google.com/spanner/docs/find-approximate-nearest-neighbors).* [↩](#cite-spanner-ann)
- **[azure-ai-search-vector-index]** **<a id="ref-azure-ai-search-vector-index"></a>Microsoft.** *[Create a Vector Index - Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-create-index).* [↩](#cite-azure-ai-search-vector-index)
- **[cosmos-db-vector]** **<a id="ref-cosmos-db-vector"></a>Microsoft.** *[Vector search in Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-search).* [↩](#cite-cosmos-db-vector)
- **[milvus-in-memory]** **<a id="ref-milvus-in-memory"></a>Milvus.** *[In-memory Index | Milvus Documentation](https://milvus.io/docs/index.md).* [↩](#cite-milvus-in-memory)
- **[milvus-scann]** **<a id="ref-milvus-scann"></a>Milvus.** *[SCANN | Milvus Documentation](https://milvus.io/docs/scann.md).* [↩](#cite-milvus-scann)
- **[weaviate-vector-index]** **<a id="ref-weaviate-vector-index"></a>Weaviate.** *[Vector index | Weaviate Documentation](https://docs.weaviate.io/weaviate/config-refs/indexing/vector-index).* [↩](#cite-weaviate-vector-index)
- **[singlestore-vector-indexing]** **<a id="ref-singlestore-vector-indexing"></a>SingleStore.** *[Vector Indexing | SingleStore Documentation](https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/).* [↩](#cite-singlestore-vector-indexing)
- **[lancedb-vector-indexes]** **<a id="ref-lancedb-vector-indexes"></a>LanceDB.** *[Vector Indexes | LanceDB Documentation](https://docs.lancedb.com/indexing/vector-index).* [↩](#cite-lancedb-vector-indexes)
- **[pgvector-github]** **<a id="ref-pgvector-github"></a>pgvector.** *[pgvector: Open-source vector similarity search for PostgreSQL](https://github.com/pgvector/pgvector).* GitHub. [↩](#cite-pgvector-github)
- **[hubness-radovanovic]** **<a id="ref-hubness-radovanovic"></a>Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010).** *[Hubs in space: Popular nearest neighbors in high-dimensional data](https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf).* Journal of Machine Learning Research (JMLR). [↩](#cite-hubness-radovanovic)
- **[elasticsearch-knn]** **<a id="ref-elasticsearch-knn"></a>Elasticsearch.** *[k-nearest neighbor (kNN) search | Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).* [↩](#cite-elasticsearch-knn)
- **[databricks-mosaic-ai]** **<a id="ref-databricks-mosaic-ai"></a>Databricks.** *[Mosaic AI Vector Search | Databricks Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html).* [↩](#cite-databricks-mosaic-ai)
- **[elasticsearch-vector]** **<a id="ref-elasticsearch-vector"></a>Elasticsearch.** *[Vector search in Elasticsearch | Elastic Docs](https://www.elastic.co/docs/solutions/search/vector).* [↩](#cite-elasticsearch-vector)
- **[splade-paper]** **<a id="ref-splade-paper"></a>Formal, T., et al. (2021).** *[SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720).* Special Interest Group on Information Retrieval Conference (SIGIR). [↩](#cite-splade-paper)
- **[jl-lemma-proof]** **<a id="ref-jl-lemma-proof"></a>Dasgupta, S., & Gupta, A. (2003).** *[An elementary proof of a theorem of Johnson and Lindenstrauss](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf).* Random Structures & Algorithms. [↩](#cite-jl-lemma-proof)
- **[rrf-paper]** **<a id="ref-rrf-paper"></a>Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).** *[Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114).* Special Interest Group on Information Retrieval Conference (SIGIR). [↩](#cite-rrf-paper)
- **[bm25-framework]** **<a id="ref-bm25-framework"></a>Robertson, S., & Zaragoza, H. (2009).** *[The Probabilistic Relevance Framework: BM25 and Beyond](https://www.emerald.com/ftinr/article/4/1-2/1/1326508/The-Probabilistic-Relevance-Framework-BM25-and).* Foundations and Trends in Information Retrieval. [↩](#cite-bm25-framework)
- **[charikar-simhash]** **<a id="ref-charikar-simhash"></a>Charikar, M. S. (2002).** *[Similarity estimation techniques from rounding algorithms](https://dl.acm.org/doi/10.1145/509907.509965).* Symposium on Theory of Computing (STOC). [↩](#cite-charikar-simhash)
- **[descartes-01ai]** **<a id="ref-descartes-01ai"></a>01.AI.** *[Descartes (ANN engine / library)](https://github.com/01-ai/Descartes).* GitHub repository.
- **[qsgngt]** **<a id="ref-qsgngt"></a>QSGNGT authors.** *[qsgngt (NGT-qg/Efanna/SSG-based ANN implementation)](https://github.com/WPJiang/HWTL_SDU-ANNS).* Project repository / documentation.
