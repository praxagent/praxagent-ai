class ArticleKnowledgeGraphInline {
    constructor() {
        this.container = null;
        this.network = null;
        this.magnifyNetwork = null;
        this.nodes = null;
        this.edges = null;
        this.concepts = [];
        this.relationships = [];
        this.options = {};
    }

    setup() {
        // Check if vis.js is loaded
        if (typeof vis === 'undefined') {
            console.warn('vis.js not loaded, hiding knowledge graph');
            this.hideGraph();
            return;
        }

        // Check if article data exists
        if (!window.articleKnowledgeGraph) {
            console.warn('No article knowledge graph data found, hiding knowledge graph');
            this.hideGraph();
            return;
        }

        this.container = document.getElementById('knowledge-graph-inline');
        if (!this.container) {
            console.warn('Knowledge graph container not found');
            return;
        }

        this.concepts = window.articleKnowledgeGraph.concepts;
        this.relationships = window.articleKnowledgeGraph.relationships;

        if (this.concepts.length === 0) {
            this.hideGraph();
            return;
        }

        // Add loading animation class
        const graphInline = document.querySelector('.knowledge-graph-inline');
        if (graphInline) {
            graphInline.classList.add('loading');
        }
        
        this.createGraph();
    }

    hideGraph() {
        const graphContainer = document.querySelector('.knowledge-graph-inline');
        if (graphContainer) {
            graphContainer.style.display = 'none';
        }
    }

    getTextColor() {
        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return isDarkMode ? '#f0f0f0' : '#1a1a1a';
    }
    
    getEdgeTextColor() {
        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return isDarkMode ? '#e5e5e5' : '#000000';
    }

    createGraph() {
        if (!this.container) return;

        // Create nodes from concepts - start them closer together
        const nodes = this.concepts.map((concept, index) => {
            // Position nodes in a tight circle initially
            const angle = (index / this.concepts.length) * 2 * Math.PI;
            const radius = 80; // Small radius for close start
            return {
                id: concept.id,
                label: concept.label,
                color: concept.color,
                font: {
                    color: this.getTextColor(),
                    size: 52
                },
                size: 65,
                title: concept.description || concept.label,
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius
            };
        });

        // Create edges from relationships
        const edges = this.relationships.map(rel => ({
            from: rel.from,
            to: rel.to,
            label: rel.label,
            color: {
                color: 'rgba(100, 116, 139, 0.4)',
                highlight: 'rgba(100, 116, 139, 0.8)'
            },
            font: {
                color: this.getEdgeTextColor(),
                size: 54,
                align: 'middle',
                strokeWidth: 1,
                strokeColor: 'rgba(0, 0, 0, 0.3)'
            },
            smooth: {
                enabled: true,
                type: 'cubicBezier',
                roundness: 0.2
            },
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 1.2
                }
            }
        }));

        this.nodes = new vis.DataSet(nodes);
        this.edges = new vis.DataSet(edges);

        this.options = {
            nodes: {
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.3)',
                    size: 10,
                    x: 2,
                    y: 2
                },
                font: {
                    color: this.getTextColor(),
                    size: 52,
                    face: 'Inter, system-ui, sans-serif'
                },
                size: 65
            },
            edges: {
                width: 4.5,
                font: {
                    color: this.getEdgeTextColor(),
                    size: 54,
                    align: 'middle',
                    strokeWidth: 1,
                    strokeColor: 'rgba(0, 0, 0, 0.3)'
                },
                smooth: {
                    enabled: true,
                    type: 'cubicBezier',
                    roundness: 0.2
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 1.2
                    }
                }
            },
            physics: {
                forceAtlas2Based: {
                    gravitationalConstant: -200, // Moderate repulsion between nodes
                    centralGravity: 0.002, // Slight central pull to keep compact
                    springLength: 500, // Longer edges - twice the text length
                    springConstant: 0.03, // Responsive springs
                    damping: 0.15, // Some damping to reduce jitter
                    avoidOverlap: 0.5 // Good overlap avoidance
                },
                solver: 'forceAtlas2Based',
                stabilization: {
                    enabled: true,
                    iterations: 100, // More iterations for stable start
                    updateInterval: 50,
                    fit: true
                },
                adaptiveTimestep: true,
                timestep: 0.4,
                maxVelocity: 30,
                minVelocity: 0.5
            },
            interaction: {
                hover: true,
                hoverConnectedEdges: true,
                selectConnectedEdges: false,
                dragNodes: true, // Re-enable drag with loose physics
                dragView: true,
                zoomView: false, // Disable scroll to zoom
                multiselect: false
            },
            layout: {
                improvedLayout: true
            }
        };

        this.network = new vis.Network(this.container, {
            nodes: this.nodes,
            edges: this.edges
        }, this.options);

        // Add boundary constraints
        this.addBoundaryConstraints();
        
        // Fit view after stabilization and remove loading animation
        this.network.once('stabilizationIterationsDone', () => {
            this.resetView();
            
            // Remove loading animation after graph is ready
            setTimeout(() => {
                const graphInline = document.querySelector('.knowledge-graph-inline');
                if (graphInline) {
                    graphInline.classList.remove('loading');
                }
            }, 2600); // Wait for full animation to complete
        });

        // Update text colors on theme change
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addListener(() => {
                this.updateTextColors();
            });
        }
    }



    resetView() {
        if (!this.network) return;
        
        // Fit all nodes in view with animation
        this.network.fit({
            animation: {
                duration: 500,
                easingFunction: 'easeInOutQuad'
            }
        });
        
        // Zoom out for better overview with extra buffer
        setTimeout(() => {
            const currentScale = this.network.getScale();
            const zoomedOutScale = Math.min(currentScale * 0.6, 0.8); // Even more zoomed out for buffer
            this.network.moveTo({
                scale: zoomedOutScale,
                animation: {
                    duration: 300,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }, 600);
    }

    updateTextColors() {
        if (!this.nodes || !this.edges) return;
        
        const textColor = this.getTextColor();
        
        // Update all nodes
        const updateNodes = this.concepts.map(concept => ({
            id: concept.id,
            font: {
                color: textColor,
                size: 52
            }
        }));
        
        // Update all edges
        const updateEdges = this.relationships.map(rel => ({
            from: rel.from,
            to: rel.to,
            font: {
                color: this.getEdgeTextColor(),
                size: 54,
                align: 'middle',
                strokeWidth: 1,
                strokeColor: 'rgba(0, 0, 0, 0.3)'
            }
        }));
        
        this.nodes.update(updateNodes);
        this.edges.update(updateEdges);
    }
    
    addBoundaryConstraints() {
        // Add smooth reverse gravitational forces from walls
        this.network.on('beforeDrawing', () => {
            const nodeIds = this.nodes.getIds();
            
            nodeIds.forEach(nodeId => {
                const position = this.network.getPositions([nodeId])[nodeId];
                if (position) {
                    // Wall boundaries with extra top buffer
                    const bounds = {
                        left: -200,
                        right: 200,
                        top: -120, // More buffer at top
                        bottom: 160
                    };
                    
                    // Reverse gravitational forces from each wall
                    let velocityX = 0;
                    let velocityY = 0;
                    
                    // Gravitational force strength
                    const wallGravity = 8000;
                    
                    // Force from left wall (repulsive)
                    const leftDist = Math.max(5, position.x - bounds.left);
                    if (leftDist < 100) {
                        velocityX += wallGravity / (leftDist * leftDist);
                    }
                    
                    // Force from right wall (repulsive)
                    const rightDist = Math.max(5, bounds.right - position.x);
                    if (rightDist < 100) {
                        velocityX -= wallGravity / (rightDist * rightDist);
                    }
                    
                    // Force from top wall (repulsive)
                    const topDist = Math.max(5, position.y - bounds.top);
                    if (topDist < 100) {
                        velocityY += wallGravity / (topDist * topDist);
                    }
                    
                    // Force from bottom wall (repulsive)
                    const bottomDist = Math.max(5, bounds.bottom - position.y);
                    if (bottomDist < 100) {
                        velocityY -= wallGravity / (bottomDist * bottomDist);
                    }
                    
                    // Apply smooth velocity changes
                    if (Math.abs(velocityX) > 0.5 || Math.abs(velocityY) > 0.5) {
                        this.network.moveNode(nodeId, 
                            position.x + velocityX * 0.002,
                            position.y + velocityY * 0.002
                        );
                    }
                }
            });
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.articleKnowledgeGraphInline = new ArticleKnowledgeGraphInline();
    window.articleKnowledgeGraphInline.setup();
});