"""
群居 (Qunju) - 數字生命可視化系統
Digital Life Visualization System
"""

import tkinter as tk
from tkinter import ttk
import math
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

class Visualizer:
    """Main visualization class for the digital life simulation"""
    
    def __init__(self, environment, width: int = 1200, height: int = 800):
        self.env = environment
        self.width = width
        self.height = height
        self.life_icons = {}  # Store references to life visual elements
        self.voice_indicators = {}  # Store voice visualization elements
        self.selected_life = None
        self.cell_size = 40  # Size of each map cell in pixels
        self.offset_x = 200  # Initial offset to center the view
        self.offset_y = 200
        self.zoom = 1.0
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("群居 (Qunju) - 數字生命模擬系統")
        self.root.geometry(f"{width}x{height}")
        
        # Create main frames
        self.create_main_frames()
        
        # Color schemes
        self.colors = {
            'sea': '#1E90FF', 'river': '#87CEEB', 'grassland': '#7CFC00',
            'snowland': '#F5F5F5', 'snow-mountain': '#E6E6FA', 'highland': '#32CD32',
            'hill': '#9ACD32', 'cave': '#A9A9A9', 'mountain': '#808080',
            'basin': '#FFD700', 'desert': '#F4A460', 'rocky': '#D2B48C',
            'life': '#FF6B6B', 'selected_life': '#FF0000',
            'voice': '#FFFF00', 'energy_low': '#FF4500', 'energy_high': '#00FF00',
            'grid': '#E0E0E0'
        }
        
        self.setup_canvas()
        self.setup_control_panel()
        
        self.simulation_running = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
    def create_main_frames(self):
        """Create the main layout frames"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls and info
        self.left_frame = ttk.Frame(self.main_frame, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.left_frame.pack_propagate(False)
        
        # Right panel for visualization
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def setup_canvas(self):
        """Setup the main visualization canvas"""
        # Create canvas with scrollbars
        self.canvas_frame = ttk.Frame(self.right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Calculate canvas size based on environment
        canvas_width = self.env.width * self.cell_size + 400
        canvas_height = self.env.height * self.cell_size + 400
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='white', 
                               width=800, height=600,
                               scrollregion=(0, 0, canvas_width, canvas_height))
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, 
                                   command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL,
                                   command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set,
                            xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
    def setup_control_panel(self):
        """Setup the control and information panel"""
        # Simulation controls
        control_frame = ttk.LabelFrame(self.left_frame, text="模擬控制", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="開始模擬", 
                                     command=self.toggle_simulation)
        self.start_button.pack(fill=tk.X)
        
        ttk.Button(control_frame, text="單步執行", 
                  command=self.step_simulation).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="居中視圖", 
                  command=self.center_view).pack(fill=tk.X, pady=5)
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="速度:").pack(side=tk.LEFT)
        self.speed_scale = ttk.Scale(speed_frame, from_=1, to=10, 
                                    orient=tk.HORIZONTAL)
        self.speed_scale.set(3)
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.left_frame, text="統計信息", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=8, width=35)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Selected life info frame
        self.life_info_frame = ttk.LabelFrame(self.left_frame, text="生命信息", padding=10)
        self.life_info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.life_info_text = tk.Text(self.life_info_frame, height=15, width=35)
        self.life_info_text.pack(fill=tk.BOTH, expand=True)
        
    def world_to_canvas(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world coordinates to canvas coordinates"""
        canvas_x = world_x * self.cell_size + self.offset_x
        canvas_y = world_y * self.cell_size + self.offset_y
        return canvas_x, canvas_y
        
    def draw_map(self):
        """Draw the environment map"""
        if not hasattr(self.env, 'map') or not self.env.map:
            print("No map data available")
            return
            
        print(f"Drawing map with {len(self.env.map)}x{len(self.env.map[0])} cells")
        
        # Draw biomes first (background)
        for y, row in enumerate(self.env.map):
            for x, biome in enumerate(row):
                x1, y1 = self.world_to_canvas(x, y)
                x2, y2 = self.world_to_canvas(x + 1, y + 1)
                
                color = self.colors.get(biome.name, '#CCCCCC')
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=color, outline=self.colors['grid'], 
                                           width=1, tags="map_bg")
                
                # Add biome name for larger cells
                if self.cell_size > 25:
                    self.canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2,
                                          text=biome.name[:4], font=('Arial', 8),
                                          tags="map_text")
        
        # Draw grid lines on top
        for x in range(self.env.width + 1):
            x1, y1 = self.world_to_canvas(x, 0)
            x2, y2 = self.world_to_canvas(x, self.env.height)
            self.canvas.create_line(x1, y1, x2, y2, 
                                  fill=self.colors['grid'], tags="grid", width=1)
        
        for y in range(self.env.height + 1):
            x1, y1 = self.world_to_canvas(0, y)
            x2, y2 = self.world_to_canvas(self.env.width, y)
            self.canvas.create_line(x1, y1, x2, y2, 
                                  fill=self.colors['grid'], tags="grid", width=1)
    
    def draw_lives(self):
        """Draw all living entities - these should be above the map"""
        current_life_names = [life.name for life in self.env.lifes]
        
        # Remove visuals for lives that no longer exist
        for life_id in list(self.life_icons.keys()):
            if life_id not in current_life_names:
                self.remove_life_visuals(life_id)
        
        print(f"Drawing {len(self.env.lifes)} lives")
        
        for life in self.env.lifes:
            life_id = life.name
            
            # Convert world coordinates to canvas coordinates
            canvas_x, canvas_y = self.world_to_canvas(life.position.x, life.position.y)
            
            # Determine color based on energy
            energy_ratio = life.energy / 100.0
            if energy_ratio > 0.7:
                color = self.colors['energy_high']
            elif energy_ratio > 0.3:
                color = '#FFA500'  # Orange
            else:
                color = self.colors['energy_low']
                
            # Use different color for selected life
            if life_id == self.selected_life:
                outline_color = self.colors['selected_life']
                outline_width = 3
            else:
                outline_color = 'black'
                outline_width = 2
            
            # Calculate icon size
            icon_size = 8
            
            # Create or update life visualization
            if life_id not in self.life_icons:
                print(f"Creating new life icon for {life_id} at ({canvas_x}, {canvas_y})")
                
                # Create life icon
                life_icon = self.canvas.create_oval(
                    canvas_x - icon_size, canvas_y - icon_size,
                    canvas_x + icon_size, canvas_y + icon_size,
                    fill=color, outline=outline_color, width=outline_width,
                    tags=('life', life_id)
                )
                
                # Add name label
                name_label = self.canvas.create_text(
                    canvas_x, canvas_y - icon_size - 10,
                    text=life_id, font=('Arial', 8, 'bold'),
                    fill='black', tags=('life_label', life_id)
                )
                
                # Add energy indicator
                energy_indicator = self.canvas.create_rectangle(
                    canvas_x - icon_size, canvas_y + icon_size + 2,
                    canvas_x - icon_size + (icon_size * 2 * energy_ratio), 
                    canvas_y + icon_size + 4,
                    fill=color, outline='', tags=('life_energy', life_id)
                )
                
                self.life_icons[life_id] = {
                    'icon': life_icon,
                    'label': name_label,
                    'energy': energy_indicator
                }
            else:
                # Update existing life icon
                life_visuals = self.life_icons[life_id]
                
                self.canvas.coords(life_visuals['icon'],
                                 canvas_x - icon_size, canvas_y - icon_size,
                                 canvas_x + icon_size, canvas_y + icon_size)
                
                self.canvas.coords(life_visuals['label'],
                                 canvas_x, canvas_y - icon_size - 10)
                
                # Update energy indicator
                self.canvas.coords(life_visuals['energy'],
                                 canvas_x - icon_size, canvas_y + icon_size + 2,
                                 canvas_x - icon_size + (icon_size * 2 * energy_ratio), 
                                 canvas_y + icon_size + 4)
                
                self.canvas.itemconfig(life_visuals['icon'], 
                                     fill=color, outline=outline_color, width=outline_width)
                self.canvas.itemconfig(life_visuals['energy'], fill=color)
            
            # Ensure life elements are above map
            self.canvas.tag_raise(life_visuals['icon'])
            self.canvas.tag_raise(life_visuals['label'])
            self.canvas.tag_raise(life_visuals['energy'])
    
    def draw_voices(self):
        """Visualize voice communication"""
        from main import voices
        
        # Clear old voice indicators
        current_voice_ids = [f"{pos.x}_{pos.y}" for pos in voices.keys()]
        for voice_id in list(self.voice_indicators.keys()):
            if voice_id not in current_voice_ids:
                self.canvas.delete(self.voice_indicators[voice_id])
                del self.voice_indicators[voice_id]
        
        # Draw current voices
        for pos, message in voices.items():
            voice_id = f"{pos.x}_{pos.y}"
            
            canvas_x, canvas_y = self.world_to_canvas(pos.x, pos.y)
            
            if voice_id not in self.voice_indicators:
                # Create voice indicator
                voice_indicator = self.canvas.create_oval(
                    canvas_x - 6, canvas_y - 6,
                    canvas_x + 6, canvas_y + 6,
                    fill=self.colors['voice'], outline='black', width=1,
                    tags=('voice', voice_id)
                )
                self.voice_indicators[voice_id] = voice_indicator
            else:
                # Update existing voice indicator
                self.canvas.coords(self.voice_indicators[voice_id],
                                 canvas_x - 6, canvas_y - 6,
                                 canvas_x + 6, canvas_y + 6)
            
            # Ensure voices are on top
            self.canvas.tag_raise(self.voice_indicators[voice_id])
    
    def remove_life_visuals(self, life_id: str):
        """Remove visualization for a life that no longer exists"""
        if life_id in self.life_icons:
            visuals = self.life_icons[life_id]
            self.canvas.delete(visuals['icon'])
            self.canvas.delete(visuals['label'])
            self.canvas.delete(visuals['energy'])
            del self.life_icons[life_id]
            
            # Clear selection if this was the selected life
            if life_id == self.selected_life:
                self.selected_life = None
                self.update_life_info(None)
    
    def update_statistics(self):
        """Update the statistics display"""
        stats_text = f"時間刻: {self.env.tick}\n"
        stats_text += f"生命數量: {len(self.env.lifes)}\n"
        
        if self.env.lifes:
            alive_count = sum(1 for life in self.env.lifes if life.is_alive)
            stats_text += f"存活: {alive_count}/{len(self.env.lifes)}\n"
            stats_text += f"平均能量: {self.calculate_average_energy():.1f}\n"
            stats_text += f"平均年齡: {self.calculate_average_age():.2f}\n"
            
            if self.env.lifes:
                oldest = max(self.env.lifes, key=lambda x: x.age)
                stats_text += f"最年長: {oldest.name} ({oldest.age:.2f}歲)\n"
        else:
            stats_text += "沒有生命存在\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_life_info(self, life: Optional['life']):
        """Update detailed information for selected life"""
        if not life:
            self.life_info_text.delete(1.0, tk.END)
            self.life_info_text.insert(1.0, "點擊一個生命來查看詳細信息")
            return
            
        info_text = f"名稱: {life.name}\n"
        info_text += f"年齡: {life.age:.2f}歲\n"
        info_text += f"能量: {life.energy:.1f}\n"
        info_text += f"位置: ({life.position.x:.1f}, {life.position.y:.1f})\n"
        info_text += f"存活: {'是' if life.is_alive else '否'}\n\n"
        
        info_text += "人格特質:\n"
        for trait_name, trait in life.personality.items():
            info_text += f"  {trait_name}: {trait.value:.3f}\n"
        
        info_text += f"\n記憶數量: {len(life.mind.memory.data)}\n"
        info_text += f"概念數量: {len(life.mind.concepts)}\n"
        
        self.life_info_text.delete(1.0, tk.END)
        self.life_info_text.insert(1.0, info_text)
    
    def calculate_average_energy(self) -> float:
        """Calculate average energy of all lives"""
        if not self.env.lifes:
            return 0.0
        return sum(life.energy for life in self.env.lifes) / len(self.env.lifes)
    
    def calculate_average_age(self) -> float:
        """Calculate average age of all lives"""
        if not self.env.lifes:
            return 0.0
        return sum(life.age for life in self.env.lifes) / len(self.env.lifes)
    
    def on_canvas_click(self, event):
        """Handle canvas click events for life selection"""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        # Get clicked items
        items = self.canvas.find_closest(event.x, event.y)
        if items:
            item_tags = self.canvas.gettags(items[0])
            
            # Check if a life was clicked
            for tag in item_tags:
                if tag in self.life_icons:
                    self.selected_life = tag
                    selected_life_obj = next(
                        (life for life in self.env.lifes if life.name == tag), 
                        None
                    )
                    self.update_life_info(selected_life_obj)
                    self.draw_lives()  # Redraw to update selection highlight
                    return
    
    def on_canvas_drag(self, event):
        """Handle canvas drag for panning"""
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        
        self.offset_x += dx
        self.offset_y += dy
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        self.update_display()
    
    def on_mousewheel(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(5.0, self.zoom))
        self.update_display()
    
    def center_view(self):
        """Center the view on all lives"""
        if not self.env.lifes:
            return
            
        # Calculate center of all lives
        avg_x = sum(life.position.x for life in self.env.lifes) / len(self.env.lifes)
        avg_y = sum(life.position.y for life in self.env.lifes) / len(self.env.lifes)
        
        # Center the view
        canvas_width = 800
        canvas_height = 600
        
        self.offset_x = canvas_width / 2 - avg_x * self.cell_size
        self.offset_y = canvas_height / 2 - avg_y * self.cell_size
        
        self.update_display()
    
    def toggle_simulation(self):
        """Toggle simulation running state"""
        self.simulation_running = not self.simulation_running
        if self.simulation_running:
            self.start_button.config(text="暫停模擬")
        else:
            self.start_button.config(text="繼續模擬")
    
    def step_simulation(self):
        """Step simulation one tick"""
        if not self.simulation_running:
            self.env.mainloop()
            self.update_display()
    
    def update_display(self):
        """Update all visual elements"""
        # Clear only dynamic elements
        self.canvas.delete("life")
        self.canvas.delete("life_label")
        self.canvas.delete("life_energy")
        self.canvas.delete("voice")
        
        # Redraw everything
        self.draw_lives()
        self.draw_voices()
        self.update_statistics()
        
        # Update selected life info if still exists
        if self.selected_life:
            selected_life_obj = next(
                (life for life in self.env.lifes if life.name == self.selected_life), 
                None
            )
            if selected_life_obj:
                self.update_life_info(selected_life_obj)
            else:
                self.selected_life = None
                self.update_life_info(None)
    
    def run(self):
        """Start the visualization"""
        print("Initializing visualization...")
        
        # Draw the map once (static background)
        self.draw_map()
        
        # Draw initial lives
        self.update_display()
        
        # Center view on lives
        self.center_view()
        
        def simulation_loop():
            if self.simulation_running:
                speed = int(self.speed_scale.get())
                for _ in range(speed):
                    self.env.mainloop()
                self.update_display()
            
            self.root.after(50, simulation_loop)  # Update every 50ms
        
        # Start simulation loop
        self.root.after(100, simulation_loop)
        self.root.mainloop()