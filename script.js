// Dark Mode Theme Management
function initializeTheme() {
    let theme;
    
    // Check for saved theme preference first (user's explicit choice takes precedence)
    const savedTheme = localStorage.getItem('theme');
    
    // Check system preference
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // Debug logging
    console.log('🎨 Theme Debug:', {
        savedTheme,
        prefersDark,
        hasMatchMedia: !!window.matchMedia
    });
    
    if (savedTheme) {
        theme = savedTheme;
        console.log('📱 Using saved theme:', theme);
    } else {
        theme = prefersDark ? 'dark' : 'light';
        console.log('🖥️ Using system preference:', theme);
    }
    
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update toggle button state
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.setAttribute('aria-checked', theme === 'dark');
    }
    
    console.log('✅ Theme applied:', theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    // Apply new theme
    document.documentElement.setAttribute('data-theme', newTheme);
    
    // Save preference
    localStorage.setItem('theme', newTheme);
    
    // Update toggle button state
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.setAttribute('aria-checked', newTheme === 'dark');
    }
    
    // Dispatch custom event for other components that might need to know
    window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: newTheme } }));
}

// Initialize theme before DOM content loads to prevent flash
initializeTheme();

// Listen for system theme changes (only if user hasn't set manual preference)
if (window.matchMedia) {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', function(e) {
        console.log('🔄 System theme changed:', e.matches ? 'dark' : 'light');
        
        // Only auto-update if user hasn't manually set a preference
        const savedTheme = localStorage.getItem('theme');
        if (!savedTheme) {
            const newTheme = e.matches ? 'dark' : 'light';
            console.log('🔄 Auto-switching to:', newTheme);
            document.documentElement.setAttribute('data-theme', newTheme);
            
            const themeToggle = document.getElementById('theme-toggle');
            if (themeToggle) {
                themeToggle.setAttribute('aria-checked', newTheme === 'dark');
            }
            
            // Dispatch custom event
            window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: newTheme } }));
        } else {
            console.log('⏭️ Ignoring system change, user has manual preference:', savedTheme);
        }
    });
}

// Debug helper function (call from browser console)
window.resetThemePreference = function() {
    localStorage.removeItem('theme');
    console.log('🗑️ Theme preference cleared. Reloading...');
    location.reload();
}

// Debug helper to check current state
window.checkThemeState = function() {
    const saved = localStorage.getItem('theme');
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const current = document.documentElement.getAttribute('data-theme');
    
    console.log('🔍 Current Theme State:', {
        savedPreference: saved,
        systemPrefersDark: systemDark,
        currentTheme: current,
        hasMatchMedia: !!window.matchMedia
    });
}

// Mobile Menu Management
function initializeMobileMenu() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenuOverlay = document.getElementById('mobile-menu-overlay');
    const mobileNavLinks = document.querySelectorAll('.mobile-nav-link');
    
    if (mobileMenuButton && mobileMenuOverlay) {
        // Toggle mobile menu
        mobileMenuButton.addEventListener('click', function() {
            const isActive = mobileMenuButton.classList.contains('active');
            
            if (isActive) {
                closeMobileMenu();
            } else {
                openMobileMenu();
            }
        });
        
        // Close menu when clicking on overlay (outside content)
        mobileMenuOverlay.addEventListener('click', function(e) {
            if (e.target === mobileMenuOverlay) {
                closeMobileMenu();
            }
        });
        
        // Close menu when clicking on nav links
        mobileNavLinks.forEach(link => {
            link.addEventListener('click', function() {
                closeMobileMenu();
            });
        });
        
        // Close menu on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && mobileMenuButton.classList.contains('active')) {
                closeMobileMenu();
            }
        });
        
        // Handle window resize - close mobile menu if switching to desktop
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768 && mobileMenuButton.classList.contains('active')) {
                closeMobileMenu();
            }
        });
    }
}

function openMobileMenu() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenuOverlay = document.getElementById('mobile-menu-overlay');
    
    mobileMenuButton.classList.add('active');
    mobileMenuOverlay.classList.add('active');
    document.body.style.overflow = 'hidden'; // Prevent scrolling when menu is open
}

function closeMobileMenu() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenuOverlay = document.getElementById('mobile-menu-overlay');
    
    mobileMenuButton.classList.remove('active');
    mobileMenuOverlay.classList.remove('active');
    document.body.style.overflow = ''; // Restore scrolling
}

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    // Initialize mobile menu
    initializeMobileMenu();
    
    // Set up theme toggle event listener
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
        themeToggle.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleTheme();
            }
        });
        // Make it focusable
        themeToggle.setAttribute('tabindex', '0');
        themeToggle.setAttribute('role', 'switch');
    }
    
    // Set up mobile theme toggle as well
    const mobileThemeToggle = document.querySelector('.mobile-theme-toggle .theme-toggle');
    if (mobileThemeToggle) {
        mobileThemeToggle.addEventListener('click', function() {
            toggleTheme();
            closeMobileMenu(); // Close mobile menu after theme change
        });
        mobileThemeToggle.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleTheme();
                closeMobileMenu();
            }
        });
        // Make it focusable
        mobileThemeToggle.setAttribute('tabindex', '0');
        mobileThemeToggle.setAttribute('role', 'switch');
    }
    // Handle incoming hash navigation from other pages
    if (window.location.hash) {
        setTimeout(() => {
            const targetSection = document.querySelector(window.location.hash);
            if (targetSection) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        }, 100); // Small delay to ensure page is loaded
    }
    
    // Handle smooth scrolling for same-page navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]:not([href*="/"])');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add navbar background on scroll
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.style.background = 'var(--bg-navbar-scrolled)';
            navbar.style.boxShadow = '0 2px 20px var(--shadow-navbar)';
        } else {
            navbar.style.background = 'var(--bg-navbar)';
            navbar.style.boxShadow = 'none';
        }
    });

    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe service cards
    const serviceCards = document.querySelectorAll('.service-card');
    serviceCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
        observer.observe(card);
    });

    // Observe stats
    const stats = document.querySelectorAll('.stat');
    stats.forEach((stat, index) => {
        stat.style.opacity = '0';
        stat.style.transform = 'translateY(30px)';
        stat.style.transition = `opacity 0.6s ease ${index * 0.2}s, transform 0.6s ease ${index * 0.2}s`;
        observer.observe(stat);
    });

    // Tech items hover effect
    const techItems = document.querySelectorAll('.tech-item');
    techItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.05)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Form handling
    const contactForm = document.querySelector('.form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const submitButton = this.querySelector('.submit-button');
            const originalText = submitButton.textContent;
            
            // Show loading state
            submitButton.textContent = 'Sending...';
            submitButton.disabled = true;
            submitButton.style.background = '#94a3b8';
            
            // Simulate form submission
            setTimeout(() => {
                submitButton.textContent = 'Message Sent! ✓';
                submitButton.style.background = '#10b981';
                
                // Reset form
                setTimeout(() => {
                    this.reset();
                    submitButton.textContent = originalText;
                    submitButton.disabled = false;
                    submitButton.style.background = '#6366f1';
                }, 2000);
            }, 1500);
        });
    }

    // Add typing animation to the hero code block with syntax highlighting
    const codeContent = document.querySelector('.code-content code');
    if (codeContent) {
        // Store the original HTML with syntax highlighting
        const originalHTML = codeContent.innerHTML;
        
        // Create array of characters with their HTML context
        const chars = [];
        let tempDiv = document.createElement('div');
        tempDiv.innerHTML = originalHTML;
        
        // Extract each character while preserving span context
        function extractChars(node, currentSpan = null) {
            if (node.nodeType === Node.TEXT_NODE) {
                for (let char of node.textContent) {
                    chars.push({
                        char: char,
                        span: currentSpan ? currentSpan.cloneNode(false) : null
                    });
                }
            } else if (node.nodeType === Node.ELEMENT_NODE) {
                for (let child of node.childNodes) {
                    extractChars(child, node.tagName === 'SPAN' ? node : currentSpan);
                }
            }
        }
        
        extractChars(tempDiv);
        
        // Clear content and start typing
        codeContent.innerHTML = '';
        let i = 0;
        let currentContent = '';
        
        const typeWriter = () => {
            if (i < chars.length) {
                const charData = chars[i];
                
                if (charData.span) {
                    // Character is inside a span
                    const spanClass = charData.span.className;
                    const style = charData.span.getAttribute('style');
                    
                    if (style) {
                        currentContent += `<span style="${style}">${charData.char}</span>`;
                    } else {
                        currentContent += `<span class="${spanClass}">${charData.char}</span>`;
                    }
                } else {
                    // Plain character
                    currentContent += charData.char;
                }
                
                codeContent.innerHTML = currentContent;
                i++;
                setTimeout(typeWriter, 50);
            }
        };
        
        // Start typing animation after a delay
        setTimeout(typeWriter, 1000);
    }

    // Add parallax effect to hero section
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const heroVisual = document.querySelector('.hero-visual');
        
        if (heroVisual && scrolled < window.innerHeight) {
            heroVisual.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });

    // Animate numbers in stats
    const animateNumber = (element, target) => {
        const duration = 2000;
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            
            if (target >= 100) {
                element.textContent = Math.floor(current) + '%';
            } else if (target >= 10) {
                element.textContent = Math.floor(current) + '+';
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    };

    // Observe stats for number animation
    const statNumbers = document.querySelectorAll('.stat-number');
    const numberObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const text = entry.target.textContent;
                const number = parseInt(text.replace(/\D/g, ''));
                animateNumber(entry.target, number);
                numberObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    statNumbers.forEach(stat => {
        numberObserver.observe(stat);
    });

    // Add click effect to buttons
    const buttons = document.querySelectorAll('.primary-button, .secondary-button, .cta-button, .submit-button, .schedule-button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.style.position = 'absolute';
            ripple.style.background = 'rgba(255, 255, 255, 0.5)';
            ripple.style.borderRadius = '50%';
            ripple.style.transform = 'scale(0)';
            ripple.style.animation = 'ripple 0.6s linear';
            ripple.style.pointerEvents = 'none';
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    // Add CSS for ripple animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Easter egg: Konami code
    let konamiCode = [];
    const konami = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65]; // ↑↑↓↓←→←→BA
    
    document.addEventListener('keydown', function(e) {
        konamiCode.push(e.keyCode);
        if (konamiCode.length > 10) {
            konamiCode.shift();
        }
        
        if (konamiCode.join(',') === konami.join(',')) {
            // Trigger easter egg
            document.body.style.animation = 'rainbow 2s infinite';
            setTimeout(() => {
                document.body.style.animation = '';
            }, 10000);
            
            // Add rainbow animation
            const rainbowStyle = document.createElement('style');
            rainbowStyle.textContent = `
                @keyframes rainbow {
                    0% { filter: hue-rotate(0deg); }
                    100% { filter: hue-rotate(360deg); }
                }
            `;
            document.head.appendChild(rainbowStyle);
            
            setTimeout(() => {
                rainbowStyle.remove();
            }, 10000);
        }
    });

    // Interactive Tech Graph
    const canvas = document.getElementById('tech-graph');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        const resizeCanvas = () => {
            const container = canvas.parentElement;
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
        };
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Tech items data
        const techItems = [
            'Python', 'TensorFlow', 'PyTorch', 'AWS', 'Google Cloud', 'Terraform',
            'Packer', 'Ansible', 'Docker', 'Kubernetes', 'Helm', 'Apache Spark',
            'Apache Airflow', 'Apache Superset', 'Hive Metastore', 'Hudi/Iceberg',
            'MLflow', 'Scikit-learn', 'OpenAI', 'Anthropic/Claude', 'Gemini',
            'Hugging Face', 'LangChain', 'RAG', 'Vector Search'
        ];
        
        // Create nodes with spread-out initial positions
        const nodes = techItems.map((tech, i) => {
            const radius = Math.max(20, tech.length * 3);
            const margin = radius + 10;
            return {
                id: i,
                label: tech,
                x: margin + Math.random() * (canvas.width - 2 * margin),
                y: margin + Math.random() * (canvas.height - 2 * margin),
                vx: (Math.random() - 0.5) * 2, // Small initial velocity for natural spread
                vy: (Math.random() - 0.5) * 2,
                radius: radius,
                color: `hsl(${200 + Math.random() * 60}, 70%, ${50 + Math.random() * 20}%)`,
                targetX: 0,
                targetY: 0,
                // Smooth random walk variables
                randomPhaseX: Math.random() * Math.PI * 2,
                randomPhaseY: Math.random() * Math.PI * 2,
                randomSpeedX: 0.02 + Math.random() * 0.01,
                randomSpeedY: 0.02 + Math.random() * 0.01
            };
        });
        
        // Create connections (random network)
        const connections = [];
        nodes.forEach((node, i) => {
            const numConnections = Math.floor(Math.random() * 4) + 1;
            for (let j = 0; j < numConnections; j++) {
                const targetIndex = Math.floor(Math.random() * nodes.length);
                if (targetIndex !== i && !connections.find(c => 
                    (c.source === i && c.target === targetIndex) || 
                    (c.source === targetIndex && c.target === i)
                )) {
                    connections.push({ source: i, target: targetIndex });
                }
            }
        });
        
        let mouseX = 0, mouseY = 0;
        let isDragging = false;
        let draggedNode = null;
        let animationTime = 0;
        
        // Light pulse system
        const pulses = [];
        let lastPulseSpawn = 0;
        const pulseSpawnInterval = 800; // milliseconds
        
        // Mouse events
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
        });
        
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            // Find clicked node
            draggedNode = nodes.find(node => {
                const dx = clickX - node.x;
                const dy = clickY - node.y;
                return Math.sqrt(dx * dx + dy * dy) < node.radius;
            });
            
            if (draggedNode) {
                isDragging = true;
                canvas.style.cursor = 'grabbing';
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
            draggedNode = null;
            canvas.style.cursor = 'grab';
        });
        
        // Physics simulation
        const animate = () => {
            animationTime += 0.016; // Roughly 60fps increment
            const currentTime = Date.now();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Spawn new pulses
            if (currentTime - lastPulseSpawn > pulseSpawnInterval && connections.length > 0) {
                const randomConnection = connections[Math.floor(Math.random() * connections.length)];
                const direction = Math.random() > 0.5 ? 1 : -1; // Random direction
                pulses.push({
                    connectionIndex: connections.indexOf(randomConnection),
                    position: direction > 0 ? 0 : 1,
                    direction: direction,
                    speed: 0.008 + Math.random() * 0.004, // Random speed
                    intensity: 0.8 + Math.random() * 0.2,
                    color: `hsl(${180 + Math.random() * 40}, 80%, 70%)`, // Blue-cyan range
                    size: 3 + Math.random() * 2
                });
                lastPulseSpawn = currentTime;
            }
            
            // Update pulses
            for (let i = pulses.length - 1; i >= 0; i--) {
                const pulse = pulses[i];
                pulse.position += pulse.direction * pulse.speed;
                
                // Remove pulses that have reached the end (tighter bounds)
                if (pulse.position < -0.05 || pulse.position > 1.05) {
                    pulses.splice(i, 1);
                }
            }
            
            // Update physics
            nodes.forEach(node => {
                if (draggedNode === node) {
                    node.x = mouseX;
                    node.y = mouseY;
                    node.vx = 0;
                    node.vy = 0;
                } else {
                    // Gentle center attraction
                    const centerX = canvas.width / 2;
                    const centerY = canvas.height / 2;
                    const centerForce = 0.0003; // Much weaker to allow spreading
                    node.vx += (centerX - node.x) * centerForce;
                    node.vy += (centerY - node.y) * centerForce;
                    
                    // Mouse repulsion
                    const dx = node.x - mouseX;
                    const dy = node.y - mouseY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < 100 && distance > 0) {
                        const force = (100 - distance) * 0.01;
                        node.vx += (dx / distance) * force;
                        node.vy += (dy / distance) * force;
                    }
                    
                    // Node repulsion (extremely strong and wide range)
                    nodes.forEach(otherNode => {
                        if (node !== otherNode) {
                            const dx = node.x - otherNode.x;
                            const dy = node.y - otherNode.y;
                            const distance = Math.sqrt(dx * dx + dy * dy);
                            const minDistance = node.radius + otherNode.radius + 120; // Increased from 80
                            if (distance < minDistance && distance > 0) {
                                const force = (minDistance - distance) * 0.025; // Increased from 0.015
                                node.vx += (dx / distance) * force;
                                node.vy += (dy / distance) * force;
                            }
                        }
                    });
                    
                    // Spring forces from connections (longer edges)
                    connections.forEach(conn => {
                        if (conn.source === node.id || conn.target === node.id) {
                            const other = nodes[conn.source === node.id ? conn.target : conn.source];
                            const dx = other.x - node.x;
                            const dy = other.y - node.y;
                            const distance = Math.sqrt(dx * dx + dy * dy);
                            const idealDistance = 180; // Increased from 120 for longer edges
                            const springForce = (distance - idealDistance) * 0.005; // Gentler spring force
                            
                            if (distance > 0) {
                                node.vx += (dx / distance) * springForce;
                                node.vy += (dy / distance) * springForce;
                            }
                        }
                    });
                    
                    // Smooth random walk movement using sine waves
                    const randomWalkStrength = 0.08;
                    node.randomPhaseX += node.randomSpeedX;
                    node.randomPhaseY += node.randomSpeedY;
                    
                    const smoothRandomX = Math.sin(node.randomPhaseX) * randomWalkStrength;
                    const smoothRandomY = Math.sin(node.randomPhaseY) * randomWalkStrength;
                    
                    node.vx += smoothRandomX;
                    node.vy += smoothRandomY;
                    
                    // Apply velocity with damping
                    node.vx *= 0.9;
                    node.vy *= 0.9;
                    node.x += node.vx;
                    node.y += node.vy;
                    
                    // Strict boundary constraints with bounce-back
                    const margin = 5; // Extra margin from edges
                    if (node.x <= node.radius + margin) {
                        node.x = node.radius + margin;
                        node.vx = Math.abs(node.vx) * 0.8; // Bounce back
                    }
                    if (node.x >= canvas.width - node.radius - margin) {
                        node.x = canvas.width - node.radius - margin;
                        node.vx = -Math.abs(node.vx) * 0.8; // Bounce back
                    }
                    if (node.y <= node.radius + margin) {
                        node.y = node.radius + margin;
                        node.vy = Math.abs(node.vy) * 0.8; // Bounce back
                    }
                    if (node.y >= canvas.height - node.radius - margin) {
                        node.y = canvas.height - node.radius - margin;
                        node.vy = -Math.abs(node.vy) * 0.8; // Bounce back
                    }
                }
            });
            
            // Draw connections
            ctx.strokeStyle = 'rgba(8, 145, 178, 0.3)';
            ctx.lineWidth = 2;
            connections.forEach(conn => {
                const sourceNode = nodes[conn.source];
                const targetNode = nodes[conn.target];
                
                ctx.beginPath();
                ctx.moveTo(sourceNode.x, sourceNode.y);
                ctx.lineTo(targetNode.x, targetNode.y);
                ctx.stroke();
            });
            
            // Draw light pulses
            pulses.forEach(pulse => {
                const conn = connections[pulse.connectionIndex];
                if (conn) {
                    const sourceNode = nodes[conn.source];
                    const targetNode = nodes[conn.target];
                    
                    // Calculate pulse position along the edge
                    const pulseX = sourceNode.x + (targetNode.x - sourceNode.x) * pulse.position;
                    const pulseY = sourceNode.y + (targetNode.y - sourceNode.y) * pulse.position;
                    
                    // Draw glowing pulse
                    ctx.save();
                    
                    // Outer glow (reduced size)
                    const glowGradient = ctx.createRadialGradient(
                        pulseX, pulseY, 0,
                        pulseX, pulseY, pulse.size * 2
                    );
                    glowGradient.addColorStop(0, pulse.color.replace('70%', '60%'));
                    glowGradient.addColorStop(0.5, pulse.color.replace('70%', '20%'));
                    glowGradient.addColorStop(1, 'transparent');
                    
                    ctx.fillStyle = glowGradient;
                    ctx.beginPath();
                    ctx.arc(pulseX, pulseY, pulse.size * 2, 0, Math.PI * 2);
                    ctx.fill();
                    
                    // Inner bright core
                    const coreGradient = ctx.createRadialGradient(
                        pulseX, pulseY, 0,
                        pulseX, pulseY, pulse.size
                    );
                    coreGradient.addColorStop(0, '#ffffff');
                    coreGradient.addColorStop(0.7, pulse.color);
                    coreGradient.addColorStop(1, pulse.color.replace('70%', '30%'));
                    
                    ctx.fillStyle = coreGradient;
                    ctx.beginPath();
                    ctx.arc(pulseX, pulseY, pulse.size, 0, Math.PI * 2);
                    ctx.fill();
                    
                    ctx.restore();
                }
            });
            
            // Draw nodes
            nodes.forEach(node => {
                // Draw node circle
                ctx.beginPath();
                ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                
                // Gradient fill
                const gradient = ctx.createRadialGradient(
                    node.x - node.radius/3, node.y - node.radius/3, 0,
                    node.x, node.y, node.radius
                );
                gradient.addColorStop(0, node.color);
                gradient.addColorStop(1, node.color.replace('70%', '40%'));
                
                ctx.fillStyle = gradient;
                ctx.fill();
                
                // Border
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Text
                ctx.fillStyle = '#ffffff';
                ctx.font = `bold ${Math.max(10, node.radius/4)}px Inter`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                // Multi-line text for longer labels
                const words = node.label.split(' ');
                if (words.length > 1 && node.label.length > 12) {
                    ctx.fillText(words[0], node.x, node.y - 5);
                    ctx.fillText(words.slice(1).join(' '), node.x, node.y + 8);
                } else {
                    ctx.fillText(node.label, node.x, node.y);
                }
            });
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
});