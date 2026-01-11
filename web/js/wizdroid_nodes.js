/**
 * Wizdroid Character Nodes - Custom UI Extensions with Seasonal Effects
 * 
 * Provides custom styling, icons, and festive seasonal effects for Wizdroid nodes in ComfyUI.
 */

import { app } from "../../../scripts/app.js";

// Detect current season and festive period
function getSeasonalTheme() {
    const now = new Date();
    const month = now.getMonth() + 1; // 1-12
    const date = now.getDate();
    
    // Christmas & New Year (Dec 15 - Jan 5)
    if ((month === 12 && date >= 15) || (month === 1 && date <= 5)) {
        return {
            name: "christmas",
            season: "festive_winter",
            emoji: "🎄",
            effects: {
                glow: "0 0 20px rgba(220, 20, 60, 0.6), 0 0 40px rgba(34, 139, 34, 0.4)",
                filter: "brightness(1.1) hue-rotate(-5deg)",
                sparkle: true
            }
        };
    }
    
    // Halloween (Oct 15 - Nov 5)
    if ((month === 10 && date >= 15) || (month === 11 && date <= 5)) {
        return {
            name: "halloween",
            season: "spooky_autumn",
            emoji: "🎃",
            effects: {
                glow: "0 0 15px rgba(255, 140, 0, 0.7), 0 0 30px rgba(75, 0, 130, 0.5)",
                filter: "brightness(0.95) saturate(1.3) hue-rotate(20deg)",
                sparkle: false
            }
        };
    }
    
    // Easter (Mar 20 - Apr 25 approx)
    if ((month === 3 && date >= 20) || (month === 4 && date <= 25)) {
        return {
            name: "easter",
            season: "spring_renewal",
            emoji: "🥚",
            effects: {
                glow: "0 0 18px rgba(255, 192, 203, 0.5), 0 0 35px rgba(173, 255, 47, 0.4)",
                filter: "brightness(1.05) saturate(1.2) hue-rotate(15deg)",
                sparkle: true
            }
        };
    }
    
    // Summer solstice & bright season (Jun 15 - Sep 15)
    if ((month === 6 && date >= 15) || (month === 7) || (month === 8) || (month === 9 && date <= 15)) {
        return {
            name: "summer",
            season: "vibrant_summer",
            emoji: "☀️",
            effects: {
                glow: "0 0 25px rgba(255, 215, 0, 0.5), 0 0 45px rgba(255, 165, 0, 0.3)",
                filter: "brightness(1.15) contrast(1.1) saturate(1.3)",
                sparkle: true
            }
        };
    }
    
    // Autumn / Fall (Sep 20 - Oct 30)
    if ((month === 9 && date >= 20) || (month === 10 && date <= 30)) {
        return {
            name: "autumn",
            season: "golden_autumn",
            emoji: "🍂",
            effects: {
                glow: "0 0 20px rgba(255, 140, 0, 0.6), 0 0 35px rgba(184, 92, 23, 0.4)",
                filter: "brightness(1.08) hue-rotate(25deg) saturate(1.25)",
                sparkle: false
            }
        };
    }
    
    // Winter (Nov 6 - Dec 14)
    if ((month === 11 && date >= 6) || (month === 12 && date <= 14)) {
        return {
            name: "winter",
            season: "crystal_winter",
            emoji: "❄️",
            effects: {
                glow: "0 0 20px rgba(176, 224, 230, 0.6), 0 0 40px rgba(135, 206, 250, 0.4)",
                filter: "brightness(1.05) hue-rotate(-10deg) saturate(0.95)",
                sparkle: true
            }
        };
    }
    
    // Spring (Mar 1 - Mar 19)
    if ((month === 3 && date < 20) || (month === 4 && date > 25)) {
        return {
            name: "spring",
            season: "blooming_spring",
            emoji: "🌸",
            effects: {
                glow: "0 0 18px rgba(255, 182, 193, 0.5), 0 0 35px rgba(144, 238, 144, 0.4)",
                filter: "brightness(1.1) saturate(1.2)",
                sparkle: true
            }
        };
    }
    
    // Default spring/summer neutral
    return {
        name: "neutral",
        season: "neutral",
        emoji: "✨",
        effects: {
            glow: "0 0 15px rgba(155, 89, 182, 0.4)",
            filter: "brightness(1.0)",
            sparkle: false
        }
    };
}

// Node color scheme - consistent wizard purple theme with seasonal overlay
const WIZDROID_COLORS = {
    prompts: {
        bgcolor: "#2d1f4e",      // Dark purple
        color: "#e8d4f8",         // Light purple text
        titleColor: "#9b59b6"     // Violet title
    },
    training: {
        bgcolor: "#1f3a4e",       // Dark teal
        color: "#d4e8f8",         // Light blue text
        titleColor: "#3498db"     // Blue title
    },
    analysis: {
        bgcolor: "#3a1f4e",       // Dark magenta
        color: "#f8d4e8",         // Light pink text
        titleColor: "#e74c8c"     // Pink title
    }
};

// Get seasonal theme once
const SEASONAL_THEME = getSeasonalTheme();

// Map node types to their categories
const NODE_CATEGORIES = {
    // Prompts
    "WizdroidCharacterPrompt": "prompts",
    "WizdroidSceneGenerator": "prompts",
    "WizdroidBackground": "prompts",
    "WizdroidMetaPrompt": "prompts",
    "WizdroidPromptCombiner": "prompts",
    "WizdroidImageEdit": "prompts",
    "WizdroidMultiAngle": "prompts",
    "WizdroidContestPrompt": "prompts",
    
    // Training
    "WizdroidLoRADataset": "training",
    "WizdroidLoRATrainer": "training",
    "WizdroidLoRAValidate": "training",
    "WizdroidLoRADatasetValidator": "training",
    
    // Analysis
    "WizdroidPhotoAspect": "analysis"
};

// Create CSS for seasonal sparkle animation
function injectSeasonalStyles() {
    const styleId = "wizdroid-seasonal-styles";
    if (document.getElementById(styleId)) return;
    
    const style = document.createElement("style");
    style.id = styleId;
    
    const keyframes = SEASONAL_THEME.effects.sparkle ? `
        @keyframes wizdroid-sparkle {
            0%, 100% { opacity: 0.7; }
            50% { opacity: 1; }
        }
        
        @keyframes wizdroid-float-sparkle {
            0% {
                transform: translateY(0px) scale(1);
                opacity: 0.8;
            }
            100% {
                transform: translateY(-20px) scale(0.7);
                opacity: 0;
            }
        }
    ` : '';
    
    const seasonalRules = `
        ${keyframes}
        
        .wizdroid-node-seasonal {
            filter: ${SEASONAL_THEME.effects.filter};
            box-shadow: ${SEASONAL_THEME.effects.glow};
            ${SEASONAL_THEME.effects.sparkle ? 'animation: wizdroid-sparkle 3s ease-in-out infinite;' : ''}
        }
        
        .wizdroid-season-label {
            font-size: 11px;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 2px;
            text-align: center;
        }
    `;
    
    style.textContent = seasonalRules;
    document.head.appendChild(style);
}

// Inject seasonal CSS on load
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectSeasonalStyles);
} else {
    injectSeasonalStyles();
}

// Apply custom styling to Wizdroid nodes with seasonal effects
app.registerExtension({
    name: "Wizdroid.CustomNodes",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const category = NODE_CATEGORIES[nodeData.name];
        if (!category) return;
        
        const colors = WIZDROID_COLORS[category];
        
        // Override the node's appearance
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Apply colors
            this.bgcolor = colors.bgcolor;
            this.color = colors.titleColor;
            
            // Add seasonal badge to title
            if (!this.title.startsWith("🧙")) {
                this.title = `🧙 ${SEASONAL_THEME.emoji} ` + this.title.replace("🧙 Wizdroid: ", "");
            }
        };
    },
    
    async nodeCreated(node) {
        const category = NODE_CATEGORIES[node.comfyClass];
        if (!category) return;
        
        const colors = WIZDROID_COLORS[category];
        node.bgcolor = colors.bgcolor;
        node.color = colors.titleColor;
        
        // Apply seasonal styling effects via canvas modification
        setTimeout(() => {
            const nodeElement = document.querySelector(`[data-node-id="${node.id}"]`);
            if (nodeElement) {
                nodeElement.classList.add("wizdroid-node-seasonal");
                
                // Add seasonal label below title
                if (!nodeElement.querySelector(".wizdroid-season-label")) {
                    const seasonLabel = document.createElement("div");
                    seasonLabel.className = "wizdroid-season-label";
                    seasonLabel.textContent = `${SEASONAL_THEME.emoji} ${SEASONAL_THEME.name}`;
                    nodeElement.appendChild(seasonLabel);
                }
            }
        }, 50);
    }
});

// Add custom context menu options for Wizdroid nodes
app.registerExtension({
    name: "Wizdroid.ContextMenu",
    
    async setup() {
        const originalGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
        
        LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
            const options = originalGetNodeMenuOptions.call(this, node);
            
            // Check if this is a Wizdroid node
            if (node.comfyClass && node.comfyClass.startsWith("Wizdroid")) {
                options.unshift(null); // Separator
                options.unshift({
                    content: "🧙 Wizdroid Documentation",
                    callback: () => {
                        window.open("https://github.com/wizdroid/wizdroid-character", "_blank");
                    }
                });
            }
            
            return options;
        };
    }
});

console.log(`🧙 Wizdroid Character Nodes loaded - ${SEASONAL_THEME.emoji} ${SEASONAL_THEME.name.toUpperCase()} season active`);
