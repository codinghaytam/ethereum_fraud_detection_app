import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'var(--border)',
				input: 'var(--input)',
				ring: 'var(--ring)',
				background: 'var(--background)',
				foreground: 'var(--foreground)',
				primary: {
					DEFAULT: 'var(--primary)',
					foreground: 'var(--primary-foreground)'
				},
				secondary: {
					DEFAULT: 'var(--secondary)',
					foreground: 'var(--secondary-foreground)'
				},
				destructive: {
					DEFAULT: 'var(--destructive)',
					foreground: 'var(--destructive-foreground)'
				},
				muted: {
					DEFAULT: 'var(--muted)',
					foreground: 'var(--muted-foreground)'
				},
				accent: {
					DEFAULT: 'var(--accent)',
					foreground: 'var(--accent-foreground)'
				},
				popover: {
					DEFAULT: 'var(--popover)',
					foreground: 'var(--popover-foreground)'
				},
				card: {
					DEFAULT: 'var(--card)',
					foreground: 'var(--card-foreground)'
				},
				// Chart colors
				chart: {
					'1': 'var(--chart-1)',
					'2': 'var(--chart-2)',
					'3': 'var(--chart-3)',
					'4': 'var(--chart-4)',
					'5': 'var(--chart-5)'
				},
				// Sidebar colors
				sidebar: {
					DEFAULT: 'var(--sidebar)',
					foreground: 'var(--sidebar-foreground)',
					primary: 'var(--sidebar-primary)',
					'primary-foreground': 'var(--sidebar-primary-foreground)',
					accent: 'var(--sidebar-accent)',
					'accent-foreground': 'var(--sidebar-accent-foreground)',
					border: 'var(--sidebar-border)',
					ring: 'var(--sidebar-ring)'
				},
				// Space theme colors (keeping for backward compatibility)
				'cosmic-blue': 'hsl(var(--cosmic-blue))',
				'nebula-purple': 'hsl(var(--nebula-purple))',
				'star-white': 'hsl(var(--star-white))',
				'deep-space': 'hsl(var(--deep-space))',
				'asteroid-gray': 'hsl(var(--asteroid-gray))',
				'solar-orange': 'hsl(var(--solar-orange))',
				// Status colors
				'status-safe': 'hsl(var(--status-safe))',
				'status-warning': 'hsl(var(--status-warning))',
				'status-danger': 'hsl(var(--status-danger))',
				'status-unknown': 'hsl(var(--status-unknown))'
			},
			fontFamily: {
				'inter': ['Inter', 'system-ui', 'sans-serif'],
				'space': ['Space Grotesk', 'system-ui', 'sans-serif'],
				'mono': ['JetBrains Mono', 'monospace'],
				// New theme fonts
				'sans': ['Roboto', 'sans-serif'],
				'serif': ['Playfair Display', 'serif'],
				'fira': ['Fira Code', 'monospace']
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)',
				// Theme specific radius
				'theme-sm': 'calc(var(--radius) - 4px)',
				'theme-md': 'calc(var(--radius) - 2px)',
				'theme-lg': 'var(--radius)',
				'theme-xl': 'calc(var(--radius) + 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'radar-sweep': {
					'0%': { transform: 'rotate(0deg)' },
					'100%': { transform: 'rotate(360deg)' }
				},
				'cosmic-pulse': {
					'0%, 100%': { opacity: '1', transform: 'scale(1)' },
					'50%': { opacity: '0.8', transform: 'scale(1.05)' }
				},
				'float': {
					'0%, 100%': { transform: 'translateY(0px)' },
					'50%': { transform: 'translateY(-10px)' }
				},
				'star-twinkle': {
					'0%, 100%': { opacity: '0.3' },
					'50%': { opacity: '1' }
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'radar-sweep': 'radar-sweep 2s linear infinite',
				'cosmic-pulse': 'cosmic-pulse 2s ease-in-out infinite',
				'float': 'float 3s ease-in-out infinite',
				'star-twinkle': 'star-twinkle 2s ease-in-out infinite'
			},
			boxShadow: {
				'cosmic': 'var(--shadow-cosmic)',
				'glow': 'var(--shadow-glow)',
				'card-cosmic': 'var(--shadow-card)',
				// Theme shadows
				'theme-2xs': 'var(--shadow-2xs)',
				'theme-xs': 'var(--shadow-xs)',
				'theme-sm': 'var(--shadow-sm)',
				'theme': 'var(--shadow)',
				'theme-md': 'var(--shadow-md)',
				'theme-lg': 'var(--shadow-lg)',
				'theme-xl': 'var(--shadow-xl)',
				'theme-2xl': 'var(--shadow-2xl)'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
