


---


(IJCAI-2016) Representation Learning of Knowledge Graphs with Hierarchical Types.
Weighted Hierarchy Encoder + Soft Type Constraint

--

input data:

entity2id.txt: entity list
format: entity	entity_id

relation2id.txt: relation list
format: relation	relation_id

Note: The general hierarchical type structure in Freebase: domain/type/topic(entity)
domain is the most general sub-type

type2id.txt: type list
format: type	type_id

domain2id.txt: domain_list
format: domain	domain_id

relationType.txt: relation-specific type constraint information for type
format: relation	head type	tail type

relationDomain.txt: relation-specific type constraint information for domain
format: relation	head domain	tail domain

typeEntity.txt: type2entity list
format: type_id	entity_id_1	entity_id_2	...	entity_id_n


---