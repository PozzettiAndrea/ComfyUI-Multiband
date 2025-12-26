// SPDX-License-Identifier: MIT
// Copyright (C) 2025 ComfyUI-Multiband Contributors

/**
 * Preview Multiband Image - Dynamic channel switching without re-execution.
 * All channels are pre-rendered for all batch images; JS switches between them instantly.
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "multiband.PreviewMultibandImage",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MultibandPreview") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated?.apply(this, arguments);

            const node = this;
            this._channelNames = [];
            this._channelStats = [];      // [{global: [min,max], per_sample: [[min,max],...]}, ...]
            this._allChannelImages = [];  // [channel][batch] structure
            this._currentChannel = 0;
            this._batchSize = 1;
            this._selectedImageIndex = -1; // -1 = all images, >=0 = specific image

            // Setup channel widget with dynamic switching
            const setupChannelWidget = () => {
                const channelWidget = node.widgets?.find(w => w.name === "channel_index");
                if (!channelWidget || !node.widgets) return;

                // Move widget to end (below images)
                const idx = node.widgets.indexOf(channelWidget);
                if (idx > -1) {
                    node.widgets.splice(idx, 1);
                    node.widgets.push(channelWidget);
                }

                // Store reference
                node._channelWidget = channelWidget;

                // Override the widget's callback to switch images dynamically
                const originalCallback = channelWidget.callback;
                channelWidget.callback = function(value) {
                    if (originalCallback) originalCallback.call(this, value);

                    // Switch displayed images without re-running node
                    if (node._allChannelImages && node._allChannelImages.length > 0) {
                        const channelIdx = Math.min(value, node._allChannelImages.length - 1);
                        const channelBatchImages = node._allChannelImages[channelIdx];

                        if (channelBatchImages && channelBatchImages.length > 0) {
                            // Load ALL batch images for this channel
                            const loadPromises = channelBatchImages.map((imgInfo, batchIdx) => {
                                return new Promise((resolve) => {
                                    const imgUrl = `/view?filename=${encodeURIComponent(imgInfo.filename)}&type=${imgInfo.type}&subfolder=${encodeURIComponent(imgInfo.subfolder || '')}`;
                                    const img = new Image();
                                    img.onload = () => resolve({ img, batchIdx });
                                    img.onerror = () => resolve(null);
                                    img.src = imgUrl;
                                });
                            });

                            Promise.all(loadPromises).then((results) => {
                                // Filter out failed loads and sort by batch index
                                const loadedImages = results
                                    .filter(r => r !== null)
                                    .sort((a, b) => a.batchIdx - b.batchIdx)
                                    .map(r => r.img);

                                if (loadedImages.length > 0) {
                                    node.imgs = loadedImages;
                                    node._currentChannel = channelIdx;
                                    node._selectedImageIndex = -1;  // Reset selection on channel change
                                    node.setDirtyCanvas(true, true);
                                }
                            });
                        }
                    }
                };

                // Update max value based on available channels
                if (node._allChannelImages && node._allChannelImages.length > 0) {
                    channelWidget.options = channelWidget.options || {};
                    channelWidget.options.max = node._allChannelImages.length - 1;
                }
            };

            // Setup after widgets are created
            setTimeout(setupChannelWidget, 100);

            return r;
        };

        // Handle execution results - store all channel images
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);

            // Store all channel images for dynamic switching
            // Structure: [channel_idx][batch_idx]
            if (message?.all_channel_images && message.all_channel_images.length > 0) {
                this._allChannelImages = message.all_channel_images[0];
                const numChannels = this._allChannelImages.length;
                const batchSize = this._allChannelImages[0]?.length || 0;
                console.log("[MultibandPreview] Loaded", numChannels, "channels x", batchSize, "batch images");

                // Update widget max value
                const channelWidget = this._channelWidget;
                if (channelWidget) {
                    channelWidget.options = channelWidget.options || {};
                    channelWidget.options.max = numChannels - 1;
                }
            }

            // Store batch size
            if (message?.batch_size && message.batch_size.length > 0) {
                this._batchSize = message.batch_size[0];
            }

            // Store channel names
            if (message?.channel_names && message.channel_names.length > 0) {
                this._channelNames = message.channel_names[0];
                console.log("[MultibandPreview] Channels:", this._channelNames);
            }

            // Store channel stats (min/max)
            if (message?.channel_stats && message.channel_stats.length > 0) {
                this._channelStats = message.channel_stats[0];
                console.log("[MultibandPreview] Stats:", this._channelStats);
            }

            // Store current channel
            if (message?.current_channel && message.current_channel.length > 0) {
                this._currentChannel = message.current_channel[0];
            }
        };

        // Draw info bar using onDrawForeground - position at TOP below title
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (onDrawForeground) onDrawForeground.apply(this, arguments);

            // Only draw if we have channel data
            if (!this._channelNames || this._channelNames.length === 0) return;
            if (!this._channelStats || this._channelStats.length === 0) return;

            const ch = this._currentChannel || 0;
            const channelName = this._channelNames[ch] || `channel_${ch}`;
            const stats = this._channelStats[ch];
            if (!stats) return;

            // Determine if showing global or per-sample stats
            const selectedIdx = this._selectedImageIndex;
            let minVal, maxVal, rangeLabel;

            if (selectedIdx >= 0 && selectedIdx < (stats.per_sample?.length || 0)) {
                const ps = stats.per_sample[selectedIdx];
                minVal = ps[0];
                maxVal = ps[1];
                rangeLabel = `sample ${selectedIdx}`;
            } else {
                minVal = stats.global[0];
                maxVal = stats.global[1];
                rangeLabel = "global";
            }

            // Format numbers
            const fmt = (v) => {
                if (Math.abs(v) < 0.001 || Math.abs(v) >= 10000) {
                    return v.toExponential(2);
                }
                return v.toFixed(3);
            };

            const infoText = `ch${ch}: ${channelName}  [${fmt(minVal)}, ${fmt(maxVal)}] (${rangeLabel})`;

            ctx.save();

            // Draw at TOP of node, just below title (y=0 is top of node content area)
            const barHeight = 16;
            const y = 0;

            // Background bar - full width, semi-transparent dark
            ctx.fillStyle = "rgba(20, 30, 50, 0.9)";
            ctx.fillRect(0, y, this.size[0], barHeight);

            // Bottom border line
            ctx.strokeStyle = "rgba(80, 120, 180, 0.6)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, y + barHeight);
            ctx.lineTo(this.size[0], y + barHeight);
            ctx.stroke();

            // Text
            ctx.font = "10px monospace";
            ctx.fillStyle = "#99ccff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(infoText, this.size[0] / 2, y + barHeight / 2);

            ctx.restore();
        };

        // Track image clicks to select specific batch image
        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function(e, localPos, graphCanvas) {
            if (onMouseDown) {
                const r = onMouseDown.apply(this, arguments);
                if (r) return r;
            }

            // Check if click is in the image area
            if (this.imgs && this.imgs.length > 1 && localPos[1] > 0 && localPos[1] < this.size[1] - 80) {
                // Simple heuristic: divide image area by number of images
                const imgAreaHeight = this.size[1] - 80;
                const imgHeight = imgAreaHeight / this.imgs.length;
                const clickedIdx = Math.floor(localPos[1] / imgHeight);

                if (clickedIdx >= 0 && clickedIdx < this.imgs.length) {
                    // Toggle selection
                    if (this._selectedImageIndex === clickedIdx) {
                        this._selectedImageIndex = -1;  // Deselect
                    } else {
                        this._selectedImageIndex = clickedIdx;
                    }
                    this.setDirtyCanvas(true, true);
                }
            }
            return false;
        };
    }
});
