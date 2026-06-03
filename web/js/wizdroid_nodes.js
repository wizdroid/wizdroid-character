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
    "WizdroidCharacterPromptMale": "prompts",
    "WizdroidCharacterPromptFemale": "prompts",
    "WizdroidSceneGenerator": "prompts",
    "WizdroidBackground": "prompts",
    "WizdroidMetaPrompt": "prompts",
    "WizdroidPromptCombiner": "prompts",
    "WizdroidImageEdit": "prompts",
    "WizdroidCharacterEdit": "prompts",
    "WizdroidStyleTransferNode": "prompts",
    "WizdroidPhotographyStyleNode": "prompts",
    
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

// Dynamic Reference Images for WizdroidCharacterEdit node
app.registerExtension({
    name: "Wizdroid.DynamicRefImages",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "WizdroidCharacterEdit") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            const node = this;
            
            // Reference image entries storage
            if (!node.refImageEntries) {
                node.refImageEntries = [];
            }
            
            // Find the ref_images widget (hidden, stores JSON)
            const refImagesWidget = node.widgets?.find(w => w.name === "ref_images");
            if (refImagesWidget) {
                refImagesWidget.type = "hidden";
                refImagesWidget.computeSize = () => [0, -4];
            }
            
            // Function to sync entries to the hidden widget
            const syncToWidget = () => {
                if (!refImagesWidget) return;
                const data = node.refImageEntries
                    .filter(e => e.type !== "none")
                    .map(e => ({ index: e.index, type: e.type }));
                refImagesWidget.value = JSON.stringify(data);
                node.setDirtyCanvas(true);
            };
            
            // Function to add a new reference entry
            const addRefEntry = (index = 2, type = "none") => {
                const entryId = node.refImageEntries.length;
                const entry = { id: entryId, index, type };
                node.refImageEntries.push(entry);
                
                // Create combo widget for reference type
                const typeWidget = node.addWidget("combo", `ref_${entryId}_type`, type, (v) => {
                    entry.type = v;
                    syncToWidget();
                }, {
                    values: ["none", "clothing", "pose", "background", "lighting", "style"],
                    serialize: false
                });
                typeWidget._refEntryId = entryId;
                
                // Create number widget for image index
                const indexWidget = node.addWidget("number", `ref_${entryId}_index`, index, (v) => {
                    entry.index = Math.max(1, Math.min(9, Math.floor(v)));
                    indexWidget.value = entry.index;
                    syncToWidget();
                }, {
                    min: 1,
                    max: 9,
                    step: 1,
                    precision: 0,
                    serialize: false
                });
                indexWidget._refEntryId = entryId;
                
                entry.typeWidget = typeWidget;
                entry.indexWidget = indexWidget;
                
                syncToWidget();
                node.setSize(node.computeSize());
                return entry;
            };
            
            // Function to remove the last reference entry
            const removeLastRefEntry = () => {
                if (node.refImageEntries.length === 0) return;
                
                const entry = node.refImageEntries.pop();
                
                // Remove widgets
                const typeIdx = node.widgets.indexOf(entry.typeWidget);
                if (typeIdx !== -1) node.widgets.splice(typeIdx, 1);
                
                const indexIdx = node.widgets.indexOf(entry.indexWidget);
                if (indexIdx !== -1) node.widgets.splice(indexIdx, 1);
                
                syncToWidget();
                node.setSize(node.computeSize());
            };
            
            // Add "+" button to add reference
            node.addWidget("button", "➕ Add Reference Image", null, () => {
                const nextIndex = node.refImageEntries.length + 2; // Start from image 2
                addRefEntry(Math.min(nextIndex, 9), "none");
            }, { serialize: false });
            
            // Add "−" button to remove reference
            node.addWidget("button", "➖ Remove Last Reference", null, () => {
                removeLastRefEntry();
            }, { serialize: false });
            
            // Load existing entries from widget value
            if (refImagesWidget && refImagesWidget.value) {
                try {
                    const data = JSON.parse(refImagesWidget.value);
                    if (Array.isArray(data)) {
                        data.forEach(item => {
                            if (item.index && item.type) {
                                addRefEntry(item.index, item.type);
                            }
                        });
                    }
                } catch (e) {
                    // Parse simple format fallback
                    const str = refImagesWidget.value;
                    if (str && str.includes(':')) {
                        str.split(',').forEach(part => {
                            const [idx, type] = part.split(':');
                            if (idx && type) {
                                addRefEntry(parseInt(idx.trim()), type.trim());
                            }
                        });
                    }
                }
            }
            
            // Store functions for external access
            node.addRefEntry = addRefEntry;
            node.removeLastRefEntry = removeLastRefEntry;
            node.syncRefImages = syncToWidget;
        };
        
        // Handle serialization to save dynamic entries
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
            // Store ref entries in node properties for reload
            o.refImageEntries = this.refImageEntries?.map(e => ({
                index: e.index,
                type: e.type
            })) || [];
        };
        
        // Handle deserialization to restore dynamic entries
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(o) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            
            // Only restore from serialized property if it exists;
            // otherwise keep entries already loaded from widget value
            // (backward compat with older workflows)
            if (o.refImageEntries && Array.isArray(o.refImageEntries)) {
                // Clear existing entries to prevent duplication on reload
                while (this.refImageEntries && this.refImageEntries.length > 0) {
                    this.removeLastRefEntry();
                }
                
                // Restore entries after a short delay to ensure widgets are ready
                if (o.refImageEntries.length > 0) {
                    setTimeout(() => {
                        o.refImageEntries.forEach(entry => {
                            if (this.addRefEntry && entry.index && entry.type) {
                                this.addRefEntry(entry.index, entry.type);
                            }
                        });
                    }, 100);
                }
            }
        };
    }
});

// ──────────────────────────────────────────────
// Style Transfer Thumbnail Display
// ──────────────────────────────────────────────

// Map style names to thumbnail filenames
const STYLE_TRANSFER_THUMBNAILS = {
    "American Comic Book": "american_comic.png",
    "Manga": "manga.png",
    "Graphic Novel": "graphic_novel.png",
    "Comic Strip": "comic_strip.png",
    "Golden Age Comic": "golden_age_comic.png",
    "Ligne Claire (European Comic)": "ligne_claire.png",
    "Studio Ghibli Anime": "ghibli.png",
    "Shonen Anime": "shonen.png",
    "Shojo Anime": "shojo.png",
    "Retro Anime (80s-90s)": "retro_anime.png",
    "Cel-Shaded Anime": "cel_shaded.png",
    "Mecha Anime": "mecha.png",
    "Anime Key Visual": "key_visual.png",
    "Seinen (Dark Manga)": "seinen.png",
    "Webtoon Style": "webtoon.png",
    "Digital Painting": "digital_painting.png",
    "Cyberpunk Comic": "cyberpunk_comic.png",
    "Superhero Comic": "superhero_comic.png",
    "Dark Gothic Fantasy Comic": "dark_gothic_fantasy.png",
};

// Base URL for extension-served images
function getThumbnailUrl(filename) {
    if (!filename) return null;
    return `/extensions/wizdroid-character/images/${filename}`;
}

app.registerExtension({
    name: "Wizdroid.StyleTransferThumbnail",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "WizdroidStyleTransferNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            const node = this;

            // Find the style combo widget
            const styleWidget = node.widgets?.find(w => w.name === "style");
            if (!styleWidget) return;

            // Create thumbnail image element
            const img = document.createElement("img");
            img.style.width = "100%";
            img.style.maxWidth = "280px";
            img.style.height = "auto";
            img.style.display = "none";
            img.style.margin = "8px auto 4px auto";
            img.style.borderRadius = "8px";
            img.style.border = "2px solid rgba(155, 89, 182, 0.3)";
            img.style.objectFit = "cover";
            img.style.boxShadow = "0 2px 12px rgba(0,0,0,0.3)";
            img.alt = "Style preview";
            img.loading = "lazy";

            // Fallback: hide if image fails to load
            img.onerror = function () {
                this.style.display = "none";
            };

            // Function to update thumbnail based on selected style
            const updateThumbnail = (styleName) => {
                const filename = STYLE_TRANSFER_THUMBNAILS[styleName];
                const url = getThumbnailUrl(filename);
                if (url) {
                    img.src = url;
                    img.style.display = "block";
                } else {
                    img.style.display = "none";
                }
            };

            // Handle "Random" — pick a random thumbnail key
            const getEffectiveStyle = (val) => {
                if (val === "Random") {
                    const keys = Object.keys(STYLE_TRANSFER_THUMBNAILS);
                    return keys[Math.floor(Math.random() * keys.length)];
                }
                return val;
            };

            // Add the DOM widget to show the image
            const thumbWidget = node.addDOMWidget("thumbnail_preview", "preview", img, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return ""; },
                setValue(v) {},
            });

            // Set the display area size
            thumbWidget.computeSize = function (width) {
                return [280, 175];
            };

            // Override the original size to make room
            const origComputeSize = node.computeSize;
            node.computeSize = function (targetWidth) {
                const size = origComputeSize ? origComputeSize.call(this, targetWidth) : [300, 100];
                size[1] += 190; // Add space for the thumbnail
                return size;
            };

            // Listen for style changes
            const originalCallback = styleWidget.callback;
            styleWidget.callback = function (val) {
                if (originalCallback) originalCallback.call(this, val);
                updateThumbnail(getEffectiveStyle(val));
            };

            // Initial thumbnail load
            setTimeout(() => {
                const val = styleWidget.value || "Random";
                updateThumbnail(getEffectiveStyle(val));
            }, 200);

            // Force a resize to accommodate the thumbnail
            setTimeout(() => {
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true);
            }, 250);
        };
    }
});

// ──────────────────────────────────────────────
// Photography Style Thumbnail Display
// ──────────────────────────────────────────────

const PHOTOGRAPHY_THUMBNAILS = {
    "Film Noir": "film_noir.png",
    "Vintage Polaroid": "vintage_polaroid.png",
    "HDR Photography": "hdr.png",
    "Tilt-Shift Photography": "tilt_shift.png",
    "Macro Photography": "macro.png",
    "Long Exposure": "long_exposure.png",
    "Neon Noir": "neon_noir.png",
    "Glamour Photography": "glamour.png",
    "Cinematic Photography": "cinematic.png",
    "Silhouette": "silhouette.png",
    "Analog Film": "analog_film.png",
    "iPhone Photography": "iphone.png",
    "Dark Moody Atmosphere": "dark_moody.png",
    "Volumetric Lighting": "volumetric_lighting.png",
    "Knolling Photography": "knolling.png",
};

app.registerExtension({
    name: "Wizdroid.PhotographyThumbnail",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "WizdroidPhotographyStyleNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            const node = this;
            const styleWidget = node.widgets?.find(w => w.name === "style");
            if (!styleWidget) return;

            const img = document.createElement("img");
            img.style.width = "100%";
            img.style.maxWidth = "280px";
            img.style.height = "auto";
            img.style.display = "none";
            img.style.margin = "8px auto 4px auto";
            img.style.borderRadius = "8px";
            img.style.border = "2px solid rgba(52, 152, 219, 0.3)";
            img.style.objectFit = "cover";
            img.style.boxShadow = "0 2px 12px rgba(0,0,0,0.3)";
            img.alt = "Style preview";
            img.loading = "lazy";

            img.onerror = function () {
                this.style.display = "none";
            };

            const updateThumbnail = (styleName) => {
                const filename = PHOTOGRAPHY_THUMBNAILS[styleName];
                const url = getThumbnailUrl(filename);
                if (url) {
                    img.src = url;
                    img.style.display = "block";
                } else {
                    img.style.display = "none";
                }
            };

            const getEffectiveStyle = (val) => {
                if (val === "Random") {
                    const keys = Object.keys(PHOTOGRAPHY_THUMBNAILS);
                    return keys[Math.floor(Math.random() * keys.length)];
                }
                return val;
            };

            const thumbWidget = node.addDOMWidget("thumbnail_preview", "preview", img, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return ""; },
                setValue(v) {},
            });

            thumbWidget.computeSize = function (width) {
                return [280, 175];
            };

            const origComputeSize = node.computeSize;
            node.computeSize = function (targetWidth) {
                const size = origComputeSize ? origComputeSize.call(this, targetWidth) : [300, 100];
                size[1] += 190;
                return size;
            };

            const originalCallback = styleWidget.callback;
            styleWidget.callback = function (val) {
                if (originalCallback) originalCallback.call(this, val);
                updateThumbnail(getEffectiveStyle(val));
            };

            setTimeout(() => {
                const val = styleWidget.value || "Random";
                updateThumbnail(getEffectiveStyle(val));
            }, 200);

            setTimeout(() => {
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true);
            }, 250);
        };
    }
});

console.log(`🧙 Wizdroid Character Nodes loaded - ${SEASONAL_THEME.emoji} ${SEASONAL_THEME.name.toUpperCase()} season active`);
