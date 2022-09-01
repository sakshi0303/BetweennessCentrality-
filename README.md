

## Scenario 1 - 937 nodes


---Execution Time for sequential processing: 1.0252282619476318 seconds ---
---Execution Time for parallel processing with 4 processes: 4.831114768981934 seconds ---
---Execution Time for NX library is: 1.5700099468231201 seconds ---

This indicates that the effort to calculate the Betweeness Centrality of 937 nodes is less intensive than spawning 4 sub processes and partitioning the graph dataset into these processes and then calculating the betweeness centrality for each subset of the data

## Scenario 2 - 8941 nodes with 2 parallel process

Loaded nodes: 8941 and edges: 10000
---Execution Time for sequential processing: 149.6292109489441 seconds ---
---Execution Time for parallel processing with 2 processes: 193.12290859222412 seconds ---
Total nodes being processed: 8941
---Execution Time for NX library is: 204.27400016784668 seconds ---

## Scenario 3 - 8941 nodes with 4 parallel process

---Execution Time for sequential processing: 141.41497325897217 seconds ---
---Execution Time for parallel processing with 4 processes: 115.68572497367859 seconds ---


This indicates as the number of nodes increase the benefits of parallel processing are visible

## Scenario 4 - 8941 nodes with 10 parallel process

Loaded nodes: 8941 and edges: 10000
---Execution Time for sequential processing: 141.78956770896912 seconds ---
---Execution Time for parallel processing with 10 processes: 325.13168001174927 seconds ---
Total nodes being processed: 8941
---Execution Time for NX library is: 186.7122814655304 seconds ---

## Scenario 5 - 24425 nodes with 4 parallel process

C:\Users\kupal\OneDrive\Documents\Projects\BetweenessCentrality\module>python main.py
Loaded nodes: 24425 and edges: 30000
---Execution Time for sequential processing: 1686.650509595871 seconds ---
---Execution Time for parallel processing with 4 processes: 2299.6116347312927 seconds ---
Total nodes being processed: 24425


## NEW SCENARIOS
## Scenario 6 - Loaded nodes: 100 and edges: 564
---Execution Time for BFS: 4.767955303192139 seconds ---
---Execution Time for Bidirectional search: 3.527918577194214 seconds ---
---Execution Time for sequential processing: 0.02094292640686035 seconds ---
---Execution Time for parallel processing with 4 processes: 3.3883004188537598 seconds ---


