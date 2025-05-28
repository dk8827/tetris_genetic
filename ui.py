# ui.py
import pygame
import os
from config import (S_WIDTH, S_HEIGHT, FONT_MAP, COLOR_BG, COLOR_WHITE,
                    NN_NODE_RADIUS, NN_NODE_SPACING_Y_WITHIN_LAYER, NN_LINE_THICKNESS_MULTIPLIER,
                    NN_PADDING, NN_TITLE_COLOR, NN_BG_COLOR, NN_BORDER_COLOR, COLOR_DARK_GREY)


class MainMenu:
    def __init__(self, surface, neat_model_path):
        self.surface = surface
        self.neat_model_path = neat_model_path
        self.font_title = FONT_MAP['arial_64_bold']
        self.font_subtitle = FONT_MAP['arial_20']
        self.font_button = FONT_MAP['arial_28_bold']
        self.font_status = FONT_MAP['arial_18']
        self.clock = pygame.time.Clock()

        self.colors = {
            "background_grad_top": COLOR_BG, # (15,15,25)
            "background_grad_bottom_tint_r": int(COLOR_BG[0] * (1-0.3*0.3)), # Example gradient
            "background_grad_bottom_tint_g": int(COLOR_BG[1] * (1-0.3*0.3)),
            "background_grad_bottom_tint_b": int(COLOR_BG[2] + 0.3*15),
            "button_normal": (45,55,85), "button_hover": (60,75,115),
            "button_disabled": (40,40,50), "button_border": (80,100,140),
            "button_border_hover": (100,130,180), "button_border_disabled": (60,60,70),
            "text_normal": COLOR_WHITE, "text_disabled": (120,120,130),
            "title_color": (220,230,255), "subtitle_color": (160,170,200)
        }
        
        button_width, button_height, button_spacing = 480, 70, 25
        num_buttons = 4
        total_buttons_height = (button_height * num_buttons) + (button_spacing * (num_buttons - 1))
        self.start_y = (S_HEIGHT - total_buttons_height) // 2 + 40
        button_x = (S_WIDTH - button_width) // 2

        self.buttons_layout = {
            "train_draw": {"text": "Train AI (with Visualization)", "rect": pygame.Rect(button_x, self.start_y, button_width, button_height)},
            "train_no_draw": {"text": "Train AI (No Visualization - Faster)", "rect": pygame.Rect(button_x, self.start_y + (button_height + button_spacing), button_width, button_height)},
            "play_saved": {"text": "Play with Saved AI Model", "rect": pygame.Rect(button_x, self.start_y + 2 * (button_height + button_spacing), button_width, button_height)},
            "human_play": {"text": "Play Tetris (Human Controlled)", "rect": pygame.Rect(button_x, self.start_y + 3 * (button_height + button_spacing), button_width, button_height)}
        }
        self.model_exists = os.path.exists(self.neat_model_path)


    def _draw_button(self, name, text, rect, is_hovered, is_enabled):
        bg_color = self.colors["button_hover"] if is_hovered and is_enabled else \
                   (self.colors["button_normal"] if is_enabled else self.colors["button_disabled"])
        border_color = self.colors["button_border_hover"] if is_hovered and is_enabled else \
                       (self.colors["button_border"] if is_enabled else self.colors["button_border_disabled"])
        text_color = self.colors["text_normal"] if is_enabled else self.colors["text_disabled"]

        # Shadow
        shadow_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width, rect.height)
        shadow_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0,0,0,80), (0,0,rect.width,rect.height), border_radius=12)
        self.surface.blit(shadow_surface, shadow_rect.topleft)
        
        pygame.draw.rect(self.surface, bg_color, rect, border_radius=12)
        pygame.draw.rect(self.surface, border_color, rect, 3 if is_hovered and is_enabled else 2, border_radius=12)

        text_surface = self.font_button.render(text, True, text_color)
        self.surface.blit(text_surface, (rect.centerx - text_surface.get_width()//2, 
                                          rect.centery - text_surface.get_height()//2))

    def display(self):
        title_label = self.font_title.render('NEAT TETRIS AI', True, self.colors["title_color"])
        subtitle_label = self.font_subtitle.render('Choose your action', True, self.colors["subtitle_color"])
        
        while True:
            # Background gradient
            for y_grad in range(S_HEIGHT):
                ratio = y_grad / S_HEIGHT
                # A simple linear interpolation for gradient
                r = int(self.colors["background_grad_top"][0] * (1 - ratio) + self.colors["background_grad_bottom_tint_r"] * ratio)
                g = int(self.colors["background_grad_top"][1] * (1 - ratio) + self.colors["background_grad_bottom_tint_g"] * ratio)
                b = int(self.colors["background_grad_top"][2] * (1 - ratio) + self.colors["background_grad_bottom_tint_b"] * ratio)
                pygame.draw.line(self.surface, (r,g,b), (0, y_grad), (S_WIDTH, y_grad))

            # Title and Subtitle
            title_y_pos = self.start_y - title_label.get_height() - 80
            self.surface.blit(self.font_title.render('NEAT TETRIS AI', True, (0,0,0)), 
                              (S_WIDTH//2 - title_label.get_width()//2 + 2, title_y_pos + 2)) # Shadow
            self.surface.blit(title_label, (S_WIDTH//2 - title_label.get_width()//2, title_y_pos))
            self.surface.blit(subtitle_label, (S_WIDTH//2 - subtitle_label.get_width()//2, 
                                               title_y_pos + title_label.get_height() + 10))

            mouse_pos = pygame.mouse.get_pos()
            for name, data in self.buttons_layout.items():
                is_enabled = True
                if name == "play_saved":
                    is_enabled = self.model_exists
                self._draw_button(name, data["text"], data["rect"], data["rect"].collidepoint(mouse_pos), is_enabled)

            if not self.model_exists:
                status_msg = self.font_status.render('(No saved AI model found for "Play with Saved AI")', True, self.colors["text_disabled"])
                play_saved_rect = self.buttons_layout["play_saved"]["rect"]
                self.surface.blit(status_msg, (play_saved_rect.centerx - status_msg.get_width()//2, 
                                               play_saved_rect.bottom + 12))

            instr_surf = self.font_status.render("Press Q to quit | Click a button", True, self.colors["subtitle_color"])
            self.surface.blit(instr_surf, (S_WIDTH//2 - instr_surf.get_width()//2, S_HEIGHT - 40))

            for event in pygame.event.get():
                if event.type == pygame.QUIT: return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q: return "quit"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, data in self.buttons_layout.items():
                        if data["rect"].collidepoint(mouse_pos):
                            is_enabled = True
                            if name == "play_saved": is_enabled = self.model_exists
                            if is_enabled: return name
            
            pygame.display.update()
            self.clock.tick(60)


class NNVisualizer:
    def __init__(self, surface):
        self.surface = surface
        self.font_error = FONT_MAP['comicsans_16']
        self.font_title = FONT_MAP['comicsans_16']

    def draw(self, genome, neat_config, x_start, y_start, width, height):
        if not genome or not neat_config:
            error_label = self.font_error.render('No genome data', 1, COLOR_DARK_GREY)
            pygame.draw.rect(self.surface, NN_BG_COLOR, (x_start, y_start, width, height))
            self.surface.blit(error_label, (x_start + (width - error_label.get_width())/2, 
                                           y_start + (height - error_label.get_height())/2))
            return

        pygame.draw.rect(self.surface, NN_BG_COLOR, (x_start, y_start, width, height))
        pygame.draw.rect(self.surface, NN_BORDER_COLOR, (x_start, y_start, width, height), 2)

        title_label = self.font_title.render('Best Genome NN', 1, NN_TITLE_COLOR)
        title_x = x_start + (width - title_label.get_width()) / 2
        title_y = y_start + NN_PADDING / 3
        self.surface.blit(title_label, (title_x, title_y))

        content_y_start = title_y + title_label.get_height() + NN_PADDING / 2
        content_x_start = x_start + NN_PADDING
        content_width = width - 2 * NN_PADDING
        content_height = height - (content_y_start - y_start) - NN_PADDING

        # Node processing (simplified from original for brevity, ensure correct keys)
        # This part relies heavily on NEAT's internal structure (genome.nodes, genome.connections)
        # and config (config.genome_config.input_keys, output_keys)
        
        # Safely get node sets
        input_keys = list(getattr(neat_config.genome_config, 'input_keys', []))
        output_keys = list(getattr(neat_config.genome_config, 'output_keys', []))
        
        all_node_ids_in_genome = set(n for n in genome.nodes.keys())
        
        # Identify hidden nodes by subtracting input and output keys from all nodes present in the genome
        hidden_keys_from_genome = all_node_ids_in_genome - set(input_keys) - set(output_keys)

        # --- BEGIN DEBUG LOGS ---
        print(f"--- NNVisualizer Debug ---")
        print(f"Genome ID: {genome.key}")
        print(f"All node IDs in genome: {sorted(list(all_node_ids_in_genome))}")
        print(f"NEAT Config Input Keys: {sorted(input_keys)}")
        print(f"NEAT Config Output Keys: {sorted(output_keys)}")
        print(f"Calculated Hidden Keys: {sorted(list(hidden_keys_from_genome))}")
        # --- END DEBUG LOGS ---

        layers_for_drawing = []
        # Layer 1: Input nodes (always taken from config if they exist)
        if input_keys:
            layers_for_drawing.append(sorted(list(input_keys))) # Use all defined input_keys

        # Layer 2: Hidden nodes (calculated as before)
        if hidden_keys_from_genome:
            # Ensure these hidden keys are actually in the genome's nodes (they should be by definition)
            actual_hidden_nodes_in_genome = sorted(list(hidden_keys_from_genome.intersection(all_node_ids_in_genome)))
            if actual_hidden_nodes_in_genome: # Add only if there are such hidden nodes
                 layers_for_drawing.append(actual_hidden_nodes_in_genome)

        # Layer 3: Output nodes (must be in genome.nodes)
        if output_keys:
            actual_output_nodes_in_genome = sorted(list(set(output_keys).intersection(all_node_ids_in_genome)))
            if actual_output_nodes_in_genome: # Add only if there are such output nodes
                layers_for_drawing.append(actual_output_nodes_in_genome)
        
        layers_for_drawing = [l for l in layers_for_drawing if l] # Remove empty layers (e.g., if no hidden nodes)
        
        # --- BEGIN DEBUG LOGS ---
        print(f"Layers for drawing (node_ids): {layers_for_drawing}")
        print(f"Number of layers for drawing: {len(layers_for_drawing)}")
        print(f"--- End NNVisualizer Debug ---")
        # --- END DEBUG LOGS ---

        if not layers_for_drawing:
            no_layers_label = self.font_error.render('No layers for drawing', 1, COLOR_DARK_GREY)
            self.surface.blit(no_layers_label, (x_start + (width - no_layers_label.get_width())/2, 
                                               y_start + (height - no_layers_label.get_height())/2))
            return

        num_layers = len(layers_for_drawing)
        node_positions = {}

        # Calculate X positions for layers
        layer_x_positions = []
        if num_layers == 1:
            layer_x_positions.append(content_x_start + content_width / 2)
        else:
            spacing_x = (content_width - 2 * NN_NODE_RADIUS) / (num_layers - 1) if num_layers > 1 else 0
            for i in range(num_layers):
                layer_x_positions.append(content_x_start + NN_NODE_RADIUS + i * spacing_x)
        
        # Calculate Y positions for nodes in each layer
        for i, layer_node_ids in enumerate(layers_for_drawing):
            num_nodes_in_layer = len(layer_node_ids)
            if num_nodes_in_layer == 0: continue
            current_layer_x = layer_x_positions[i]

            total_node_height_viz = num_nodes_in_layer * (2 * NN_NODE_RADIUS)
            
            if num_nodes_in_layer > 1:
                # Calculate available space for spacing, ensuring it's not too large or small
                available_y_for_spacing = content_height - total_node_height_viz
                spacing_y_per_gap = available_y_for_spacing / (num_nodes_in_layer - 1)
                actual_spacing_y = min(spacing_y_per_gap, NN_NODE_SPACING_Y_WITHIN_LAYER * 1.5)
                actual_spacing_y = max(actual_spacing_y, NN_NODE_SPACING_Y_WITHIN_LAYER * 0.5 if num_nodes_in_layer > 1 else 0)
            else:
                actual_spacing_y = 0

            total_layer_height_on_screen = total_node_height_viz + max(0, num_nodes_in_layer - 1) * actual_spacing_y
            start_y_for_layer_nodes = content_y_start + (content_height - total_layer_height_on_screen) / 2
            start_y_for_layer_nodes = max(start_y_for_layer_nodes, content_y_start) # Ensure it starts within content area

            for node_idx, node_id in enumerate(layer_node_ids):
                y_pos = start_y_for_layer_nodes + node_idx * (2 * NN_NODE_RADIUS + actual_spacing_y) + NN_NODE_RADIUS
                node_positions[node_id] = (current_layer_x, y_pos)

        # Draw connections
        if hasattr(genome, 'connections'):
            for cg_key, cg_gene in genome.connections.items():
                if cg_gene.enabled:
                    input_node_id, output_node_id = cg_gene.key
                    pos_in, pos_out = node_positions.get(input_node_id), node_positions.get(output_node_id)
                    if pos_in and pos_out:
                        weight = cg_gene.weight
                        intensity = int(min(255, max(30, 50 + abs(weight) * 150)))
                        color = (0, intensity, 0) if weight > 0 else (intensity, 0, 0)
                        thickness = int(max(1, min(abs(weight * NN_LINE_THICKNESS_MULTIPLIER), 3)))
                        try:
                             pygame.draw.line(self.surface, color, 
                                             (int(pos_in[0]), int(pos_in[1])), 
                                             (int(pos_out[0]), int(pos_out[1])), thickness)
                        except TypeError: pass # Error in pos_in/pos_out

        # Draw nodes
        for node_id, pos in node_positions.items():
            node_color, node_border_color = (180,180,180), (50,50,50) # Default for hidden
            if node_id in input_keys: node_color, node_border_color = (80,80,220), (120,120,255) # Blueish for input
            elif node_id in output_keys: node_color, node_border_color = (80,220,80), (120,255,120) # greenish for output
            
            try:
                pygame.draw.circle(self.surface, node_color, (int(pos[0]), int(pos[1])), NN_NODE_RADIUS)
                pygame.draw.circle(self.surface, node_border_color, (int(pos[0]), int(pos[1])), NN_NODE_RADIUS, 1)
            except TypeError: pass # Error in pos