// SPDX-License-Identifier: MIT
// Copyright (C) 2025 ComfyUI-Multiband Contributors

/**
 * Preview Multiband Image - Adds channel selector dropdown.
 * Uses ComfyUI's standard image preview (grid you can click to expand).
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "multiband.PreviewMultibandImage",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MultibandPreview") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const r = onNodeCreated?.apply(this, arguments);

            // Store reference for event handler
            const node = this;
            this._channelNames = [];

            // Find existing widgets and enhance them
            const setupChannelDropdown = () => {
                // Find the channel_index widget
                const channelWidget = node.widgets?.find(w => w.name === "channel_index");
                const modeWidget = node.widgets?.find(w => w.name === "mode");

                if (!channelWidget) return;

                // Store original widget for value access
                node._channelWidget = channelWidget;
                node._modeWidget = modeWidget;
            };

            // Setup after widgets are created
            setTimeout(setupChannelDropdown, 100);

            return r;
        };

        // Handle execution results - update channel names
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            onExecuted?.apply(this, arguments);

            // Store channel names for reference
            if (message?.channel_names && message.channel_names.length > 0) {
                this._channelNames = message.channel_names[0];
                console.log("[MultibandPreview] Channels:", this._channelNames);
            }
        };
    }
});
