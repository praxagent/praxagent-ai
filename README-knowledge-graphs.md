# Knowledge Graph System

Each blog post can have its own interactive knowledge graph by including a `knowledge-graph.json` file in the post's directory.

## File Structure

```
content/posts/
├── your-post/
│   ├── index.md              # Your blog post
│   ├── knowledge-graph.json  # Concept map for this post
│   └── other-assets/         # Images, etc.
```

## Knowledge Graph Format

```json
{
  "concepts": [
    {
      "id": "unique-concept-id",
      "label": "Display Name", 
      "color": "#HEX_COLOR",
      "description": "Tooltip description"
    }
  ],
  "relationships": [
    {
      "from": "concept-id-1",
      "to": "concept-id-2", 
      "label": "relationship-verb"
    }
  ],
  "conceptMap": [
    {
      "section": "heading-id",
      "concepts": ["concept-1", "concept-2"]
    }
  ]
}
```

## Section Mapping

The `conceptMap` links article sections to concepts. When you scroll to a section, those concepts highlight in the graph.

**Section IDs** are based on your heading text:
- "Problem: Lost Context" → `"problem-lost-context"`
- "Demo of the Problem" → `"demo-of-the-problem"`  
- "Solution: Late Chunking" → `"solution-late-chunking"`

## Colors

Use distinct colors for different concept types:
- **Problems**: Red (`#EF4444`)
- **Solutions**: Green (`#10B981`) 
- **Technologies**: Blue (`#3B82F6`)
- **Processes**: Purple (`#8B5CF6`)
- **Concepts**: Orange (`#F59E0B`)

## Benefits

✅ **Per-post customization**: Each article has its own concept map  
✅ **Version control**: Changes tracked with the post  
✅ **Easy maintenance**: Edit JSON files independently  
✅ **Automatic fallback**: Posts without knowledge graphs still work  
✅ **Rich tooltips**: Detailed descriptions on hover  
✅ **Interactive navigation**: Click concepts to jump to sections  
✅ **Scroll highlighting**: Concepts glow as you read relevant sections  

## Example Usage

1. Create your blog post: `content/posts/my-topic/index.md`
2. Add knowledge graph: `content/posts/my-topic/knowledge-graph.json`
3. Build site: `hugo`
4. Enjoy interactive concept visualization!