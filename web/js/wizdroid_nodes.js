/**
 * Wizdroid Character Nodes - Custom UI Extensions
 * 
 * Provides custom styling and icons for Wizdroid nodes in ComfyUI.
 */

import { app } from "../../../scripts/app.js";

// Node color scheme - consistent wizard purple theme
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

// Apply custom styling to Wizdroid nodes
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
            
            // Add wizard badge to title
            if (!this.title.startsWith("🧙")) {
                this.title = "🧙 " + this.title.replace("🧙 Wizdroid: ", "");
            }
        };
    },
    
    async nodeCreated(node) {
        const category = NODE_CATEGORIES[node.comfyClass];
        if (!category) return;
        
        const colors = WIZDROID_COLORS[category];
        node.bgcolor = colors.bgcolor;
        node.color = colors.titleColor;
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

console.log("🧙 Wizdroid Character Nodes loaded");
