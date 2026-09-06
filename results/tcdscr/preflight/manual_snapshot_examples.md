# Manual snapshot examples (P9 prototype)

## Event 499397048878501888 (nodes in 240h window: 7)

### cutoff +1h
- included nodes: 1 (excluded 6)
- edges (child->parent): 0
- max depth in snapshot: 0
- source replies in snapshot: 0
- future leakage: False
- included first 5: 499397048878501888
- excluded first 5: 499544240746414081, 499545033104564226, 499553591137693696, 499556138304962560, 499559925501399040
- edge list (indices): []

### cutoff +6h
- included nodes: 1 (excluded 6)
- edges (child->parent): 0
- max depth in snapshot: 0
- source replies in snapshot: 0
- future leakage: False
- included first 5: 499397048878501888
- excluded first 5: 499544240746414081, 499545033104564226, 499553591137693696, 499556138304962560, 499559925501399040
- edge list (indices): []

### cutoff +24h
- included nodes: 7 (excluded 0)
- edges (child->parent): 6
- max depth in snapshot: 3
- source replies in snapshot: 3
- future leakage: False
- included first 5: 499397048878501888, 499544240746414081, 499545033104564226, 499553591137693696, 499556138304962560
- excluded first 5: (none)
- edge list (indices): [[1, 0], [2, 0], [3, 1], [4, 1], [5, 4], [6, 0]]

## Event 500421926079066112 (nodes in 240h window: 15)

### cutoff +1h
- included nodes: 12 (excluded 3)
- edges (child->parent): 11
- max depth in snapshot: 3
- source replies in snapshot: 8
- future leakage: False
- included first 5: 500421926079066112, 500422190068944898, 500425848185319425, 500426136003051521, 500427504612831232
- excluded first 5: 500445512882216960, 500446623806132225, 500449073485193216
- edge list (indices): [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 7], [9, 8], [10, 8], [11, 0]]

### cutoff +6h
- included nodes: 15 (excluded 0)
- edges (child->parent): 14
- max depth in snapshot: 3
- source replies in snapshot: 11
- future leakage: False
- included first 5: 500421926079066112, 500422190068944898, 500425848185319425, 500426136003051521, 500427504612831232
- excluded first 5: (none)
- edge list (indices): [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 7], [9, 8], [10, 8], [11, 0], [12, 0]] ...

### cutoff +24h
- included nodes: 15 (excluded 0)
- edges (child->parent): 14
- max depth in snapshot: 3
- source replies in snapshot: 11
- future leakage: False
- included first 5: 500421926079066112, 500422190068944898, 500425848185319425, 500426136003051521, 500427504612831232
- excluded first 5: (none)
- edge list (indices): [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 7], [9, 8], [10, 8], [11, 0], [12, 0]] ...

## Event 524929559909916672 (nodes in 240h window: 18)

### cutoff +1h
- included nodes: 3 (excluded 15)
- edges (child->parent): 2
- max depth in snapshot: 1
- source replies in snapshot: 2
- future leakage: False
- included first 5: 524929559909916672, 524935197587173376, 524935747389112321
- excluded first 5: 524955490091274240, 524956695949492224, 524959023754661888, 524962559884283905, 524962728671059968
- edge list (indices): [[1, 0], [2, 0]]

### cutoff +6h
- included nodes: 18 (excluded 0)
- edges (child->parent): 17
- max depth in snapshot: 2
- source replies in snapshot: 16
- future leakage: False
- included first 5: 524929559909916672, 524935197587173376, 524935747389112321, 524955490091274240, 524956695949492224
- excluded first 5: (none)
- edge list (indices): [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]] ...

### cutoff +24h
- included nodes: 18 (excluded 0)
- edges (child->parent): 17
- max depth in snapshot: 2
- source replies in snapshot: 16
- future leakage: False
- included first 5: 524929559909916672, 524935197587173376, 524935747389112321, 524955490091274240, 524956695949492224
- excluded first 5: (none)
- edge list (indices): [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0]] ...
