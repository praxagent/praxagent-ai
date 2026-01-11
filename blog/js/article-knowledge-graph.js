class ArticleKnowledgeGraph {
    constructor() {
        this.network = null;
        this.nodes = null;
        this.edges = null;
        this.container = null;
        this.currentSection = null;
        this.sectionObserver = null;
        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        if (typeof vis === 'undefined') {
            console.warn('Vis.js library not loaded - hiding article knowledge graph');
            this.hideKnowledgeGraph();
            return;
        }

        if (!window.articleKnowledgeGraph || !document.getElementById('knowledge-graph')) {
            console.warn('Article knowledge graph data or container missing - hiding knowledge graph');
            this.hideKnowledgeGraph();
            return;
        }

        if (!window.articleKnowledgeGraph.concepts || window.articleKnowledgeGraph.concepts.length === 0) {
            console.warn('No article concepts found - hiding knowledge graph');
            this.hideKnowledgeGraph();
            return;
        }

        this.container = document.getElementById('knowledge-graph');
        this.createGraph();
        this.setupScrollTracking();
        this.setupEventListeners();
    }

    hideKnowledgeGraph() {
        const sidebar = document.querySelector('.knowledge-graph-sidebar');
        if (sidebar) {
            sidebar.style.display = 'none';
        }
        
        const blogPost = document.querySelector('.blog-post .container');
        if (blogPost) {
            blogPost.style.maxWidth = '100%';
            blogPost.style.marginRight = 'auto';
        }
    }

    createGraph() {
        const data = window.articleKnowledgeGraph;
        
        // Transform concepts into nodes
        this.nodes = new vis.DataSet(data.concepts.map(concept => ({
            id: concept.id,
            label: concept.label,
            title: `<strong>${concept.label}</strong><br/>${concept.description}`,
            color: {
                background: concept.color,
                border: this.darkenColor(concept.color),
                highlight: {
                    background: concept.color,
                    border: this.darkenColor(concept.color)
                }
            },
            font: {
                color: this.getTextColor(),
                size: 28,
                face: 'Inter, system-ui, sans-serif'
            },
            size: 35,
            active: false
        })));

        // Transform relationships into edges
        this.edges = new vis.DataSet(data.relationships.map(rel => ({
            from: rel.from,
            to: rel.to,
            label: rel.label,
            color: {
                color: 'rgba(100, 116, 139, 0.4)',
                highlight: 'rgba(100, 116, 139, 0.8)'
            },
            font: {
                color: this.getTextColor(),
                size: 28,
                align: 'middle',
                strokeWidth: 3,
                strokeColor: '#ffffff'
            },
            smooth: {
                enabled: true,
                type: 'cubicBezier',
                roundness: 0.2
            },
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.8
                }
            }
        })));

        const graphData = {
            nodes: this.nodes,
            edges: this.edges
        };

        const options = {
            layout: {
                improvedLayout: true,
                hierarchical: {
                    enabled: false
                }
            },
            physics: {
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -300,
                    centralGravity: 0.002,
                    springLength: 350,
                    springConstant: 0.02,
                    damping: 0.4,
                    avoidOverlap: 0.8
                },
                maxVelocity: 20,
                timestep: 0.35,
                stabilization: {
                    enabled: true,
                    iterations: 200
                }
            },
            interaction: {
                dragNodes: true,
                dragView: true,
                zoomView: true,
                selectConnectedEdges: false,
                hover: true,
                tooltipDelay: 200
            },
            nodes: {
                borderWidth: 2,
                borderWidthSelected: 3,
                shape: 'dot'
            },
            edges: {
                width: 2,
                selectionWidth: 3,
                hoverWidth: 3
            }
        };

        this.network = new vis.Network(this.container, graphData, options);
        
        // Setup magnifying glass effect
        this.setupMagnifyingGlass();
        
        // Ensure proper initial view after stabilization
        this.network.once('stabilizationIterationsDone', () => {
            setTimeout(() => {
                this.network.fit({
                    animation: {
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            }, 100);
        });
    }

    setupScrollTracking() {
        // Add section identifiers to headings if they don't exist
        this.addSectionIds();

        // Set up Intersection Observer to track which section is in view
        const options = {
            rootMargin: '-20% 0px -20% 0px',
            threshold: 0.1
        };

        this.sectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.highlightSectionConcepts(entry.target.id);
                }
            });
        }, options);

        // Observe all section headings
        document.querySelectorAll('h2, h3').forEach(heading => {
            if (heading.id) {
                this.sectionObserver.observe(heading);
            }
        });
    }

    addSectionIds() {
        // Map common heading patterns to section IDs
        const headingMap = {
            'Problem: Lost Context in Embedding-Based Retrieval': 'problem-lost-context',
            'Demo of the Problem': 'demo-of-the-problem',
            'Solution: Late Chunking': 'solution-late-chunking',
            'Demo of Solution': 'demo-of-solution',
            'Handling Large Documents': 'handling-large-documents'
        };

        document.querySelectorAll('h2, h3').forEach(heading => {
            const text = heading.textContent.trim();
            if (headingMap[text] && !heading.id) {
                heading.id = headingMap[text];
            }
        });
    }

    highlightSectionConcepts(sectionId) {
        if (this.currentSection === sectionId) return;
        this.currentSection = sectionId;

        const data = window.articleKnowledgeGraph;
        const sectionMap = data.conceptMap.find(map => map.section === sectionId);
        
        if (!sectionMap) {
            this.resetAllConcepts();
            return;
        }

        // Reset all nodes to inactive
        const updateNodes = data.concepts.map(concept => ({
            id: concept.id,
            color: {
                background: sectionMap.concepts.includes(concept.id) ? concept.color : this.fadeColor(concept.color),
                border: this.darkenColor(concept.color)
            },
            size: sectionMap.concepts.includes(concept.id) ? 40 : 30,
            font: {
                color: this.getTextColor(),
                size: sectionMap.concepts.includes(concept.id) ? 30 : 26
            }
        }));

        this.nodes.update(updateNodes);

        // Update connection visibility
        const updateEdges = data.relationships.map(rel => ({
            id: `${rel.from}-${rel.to}`,
            color: {
                color: (sectionMap.concepts.includes(rel.from) || sectionMap.concepts.includes(rel.to)) 
                    ? 'rgba(100, 116, 139, 0.6)' 
                    : 'rgba(100, 116, 139, 0.2)'
            },
            width: (sectionMap.concepts.includes(rel.from) || sectionMap.concepts.includes(rel.to)) ? 3 : 1
        }));

        this.edges.update(updateEdges);
    }

    resetAllConcepts() {
        const data = window.articleKnowledgeGraph;
        const updateNodes = data.concepts.map(concept => ({
            id: concept.id,
            color: {
                background: concept.color,
                border: this.darkenColor(concept.color)
            },
            size: 35,
            font: {
                color: this.getTextColor(),
                size: 28
            }
        }));

        this.nodes.update(updateNodes);
    }

    getTextColor() {
        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return isDarkMode ? '#ffffff' : '#000000';
    }

    darkenColor(color) {
        const hex = color.replace('#', '');
        const r = Math.max(0, parseInt(hex.substr(0, 2), 16) - 30);
        const g = Math.max(0, parseInt(hex.substr(2, 2), 16) - 30);
        const b = Math.max(0, parseInt(hex.substr(4, 2), 16) - 30);
        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }

    fadeColor(color) {
        const hex = color.replace('#', '');
        const r = parseInt(hex.substr(0, 2), 16);
        const g = parseInt(hex.substr(2, 2), 16);
        const b = parseInt(hex.substr(4, 2), 16);
        return `rgba(${r}, ${g}, ${b}, 0.3)`;
    }

    setupEventListeners() {
        if (!this.network) return;

        // Handle concept clicks to scroll to relevant section
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                const conceptId = params.nodes[0];
                this.scrollToConceptSection(conceptId);
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.network) {
                this.network.redraw();
                // Re-fit the view after a short delay to ensure proper sizing
                setTimeout(() => {
                    this.network.fit();
                }, 100);
            }
        });

        // Handle theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
                this.updateTextColors();
            });
        }
    }

    scrollToConceptSection(conceptId) {
        const data = window.articleKnowledgeGraph;
        const relevantSection = data.conceptMap.find(map => 
            map.concepts.includes(conceptId)
        );
        
        if (relevantSection) {
            const element = document.getElementById(relevantSection.section);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    }

    updateTextColors() {
        if (!this.nodes) return;
        
        const data = window.articleKnowledgeGraph;
        const updateNodes = data.concepts.map(concept => ({
            id: concept.id,
            font: {
                color: this.getTextColor()
            }
        }));
        
        this.nodes.update(updateNodes);
    }

    setupMagnifyingGlass() {
        if (!this.network || !this.container) return;
        
        // Create magnifying glass element
        this.magnifyingGlass = document.createElement('div');
        this.magnifyingGlass.className = 'magnifying-glass';
        this.magnifyingGlass.style.cssText = `
            position: absolute;
            width: 150px;
            height: 150px;
            border: 3px solid #0891B2;
            border-radius: 50%;
            pointer-events: none;
            z-index: 1000;
            display: none;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(1px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        `;
        
        // Create container for magnified vis.js network
        this.magnifyContainer = document.createElement('div');
        this.magnifyContainer.style.cssText = `
            width: 300px;
            height: 300px;
            border-radius: 50%;
            position: absolute;
            top: -75px;
            left: -75px;
            clip-path: circle(75px at center);
        `;
        
        this.magnifyingGlass.appendChild(this.magnifyContainer);
        this.container.parentElement.appendChild(this.magnifyingGlass);
        
        // Create magnified network with same data but higher zoom
        const magnifyOptions = {
            ...this.options,
            physics: {
                enabled: false // Disable physics for performance
            },
            interaction: {
                dragNodes: false,
                dragView: false,
                zoomView: false
            },
            nodes: {
                ...this.options.nodes,
                font: {
                    ...this.options.nodes.font,
                    size: this.options.nodes.font.size * 2 // Double the font size
                },
                size: this.options.nodes.size * 1.5 // Bigger nodes
            },
            edges: {
                ...this.options.edges,
                font: {
                    ...this.options.edges.font,
                    size: this.options.edges.font.size * 1.8 // Bigger edge labels
                },
                width: this.options.edges.width * 1.5
            }
        };
        
        // Create the magnified network
        this.magnifyNetwork = new vis.Network(this.magnifyContainer, {
            nodes: this.nodes,
            edges: this.edges
        }, magnifyOptions);
        
        // Mouse move handler
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Position magnifying glass
            this.magnifyingGlass.style.left = (x - 75) + 'px';
            this.magnifyingGlass.style.top = (y - 75) + 'px';
            this.magnifyingGlass.style.display = 'block';
            
            // Update magnified network view
            this.updateMagnifiedView(x, y);
        });
        
        // Mouse enter
        this.container.addEventListener('mouseenter', () => {
            this.container.style.cursor = 'none';
            this.magnifyingGlass.style.display = 'block';
        });
        
        // Mouse leave
        this.container.addEventListener('mouseleave', () => {
            this.container.style.cursor = 'default';
            this.magnifyingGlass.style.display = 'none';
        });
    }
    
    updateMagnifiedView(mouseX, mouseY) {
        if (!this.magnifyNetwork || !this.network) return;
        
        // Get the current view position and scale of main network
        const mainPosition = this.network.getViewPosition();
        const mainScale = this.network.getScale();
        
        // Convert mouse position to network coordinates
        const containerRect = this.container.getBoundingClientRect();
        const networkPos = this.network.DOMtoCanvas({
            x: mouseX,
            y: mouseY
        });
        
        // Set magnified network to focus on the area under cursor
        // with higher scale (zoomed in)
        const magnifiedScale = mainScale * 3; // 3x zoom
        
        this.magnifyNetwork.moveTo({
            position: {
                x: networkPos.x,
                y: networkPos.y
            },
            scale: magnifiedScale,
            animation: false // No animation for smooth tracking
        });
    }

    resetView() {
        if (!this.network) return;
        
        // Fit all nodes in view with animation
        this.network.fit({
            animation: {
                duration: 600,
                easingFunction: 'easeInOutQuad'
            }
        });
        
        // Ensure reasonable zoom level
        setTimeout(() => {
            const scale = this.network.getScale();
            if (scale > 1.2 || scale < 0.5) {
                this.network.moveTo({
                    scale: Math.min(Math.max(scale, 0.8), 1.2),
                    animation: {
                        duration: 400,
                        easingFunction: 'easeInOutQuad'
                    }
                });
            }
        }, 650);
    }

    destroy() {
        if (this.sectionObserver) {
            this.sectionObserver.disconnect();
        }
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
    }
}

// Toggle function for collapsing/expanding the sidebar
function toggleKnowledgeGraph() {
    const sidebar = document.getElementById('knowledge-graph-sidebar');
    const blogPost = document.querySelector('.blog-post .container');
    
    if (sidebar && blogPost) {
        const isCollapsed = sidebar.classList.contains('collapsed');
        
        if (isCollapsed) {
            // Expanding
            sidebar.classList.remove('collapsed');
            blogPost.classList.remove('expanded');
            localStorage.setItem('knowledgeGraphCollapsed', 'false');
        } else {
            // Collapsing
            sidebar.classList.add('collapsed');
            blogPost.classList.add('expanded');
            localStorage.setItem('knowledgeGraphCollapsed', 'true');
        }
        
        // Redraw and reset view after transition completes
        setTimeout(() => {
            if (articleKnowledgeGraph && articleKnowledgeGraph.network && !sidebar.classList.contains('collapsed')) {
                // Redraw the network and reset view
                articleKnowledgeGraph.network.redraw();
                articleKnowledgeGraph.resetView();
            }
        }, 450);
    }
}

// Initialize the article knowledge graph
let articleKnowledgeGraph;
document.addEventListener('DOMContentLoaded', () => {
    articleKnowledgeGraph = new ArticleKnowledgeGraph();
    
    // Restore collapsed state from localStorage
    const isCollapsed = localStorage.getItem('knowledgeGraphCollapsed') === 'true';
    if (isCollapsed) {
        const sidebar = document.getElementById('knowledge-graph-sidebar');
        const blogPost = document.querySelector('.blog-post .container');
        if (sidebar) {
            sidebar.classList.add('collapsed');
        }
        if (blogPost) {
            blogPost.classList.add('expanded');
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (articleKnowledgeGraph) {
        articleKnowledgeGraph.destroy();
    }
});