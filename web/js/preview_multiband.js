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
            this._allChannelImages = [];  // [channel][batch] structure
            this._currentChannel = 0;
            this._batchSize = 1;

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

            // Store current channel
            if (message?.current_channel && message.current_channel.length > 0) {
                this._currentChannel = message.current_channel[0];
            }
        };
    }
});
