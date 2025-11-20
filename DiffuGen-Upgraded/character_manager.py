"""
Character Consistency Manager for DiffuGen
Manages character descriptions and reference images for consistent character generation.
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import shutil


class CharacterManager:
    """Manages character definitions for consistent image generation."""

    def __init__(self, characters_dir: Optional[str] = None):
        """
        Initialize the character manager.

        Args:
            characters_dir: Directory to store character data. Defaults to ./characters/
        """
        self.characters_dir = Path(characters_dir or os.path.join(os.path.dirname(__file__), "characters"))
        self.characters_dir.mkdir(exist_ok=True)

        self.characters_file = self.characters_dir / "characters.json"
        self.reference_images_dir = self.characters_dir / "reference_images"
        self.reference_images_dir.mkdir(exist_ok=True)

        self.characters = self._load_characters()

    def _load_characters(self) -> Dict:
        """Load characters from JSON file."""
        if self.characters_file.exists():
            try:
                with open(self.characters_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load characters file: {e}")
                return {}
        return {}

    def _save_characters(self):
        """Save characters to JSON file."""
        try:
            with open(self.characters_file, 'w', encoding='utf-8') as f:
                json.dump(self.characters, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving characters file: {e}")

    def add_character(self,
                     name: str,
                     description: str,
                     reference_image: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     negative_prompt: Optional[str] = None) -> Dict:
        """
        Add or update a character.

        Args:
            name: Unique character name/identifier
            description: Detailed description of the character's appearance
            reference_image: Path to reference image (will be copied to characters dir)
            tags: Additional tags for categorization
            negative_prompt: Default negative prompt for this character

        Returns:
            The character data dictionary
        """
        character_data = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "negative_prompt": negative_prompt or "",
            "reference_image": None
        }

        # Handle reference image
        if reference_image and os.path.exists(reference_image):
            # Copy image to reference_images directory
            ext = os.path.splitext(reference_image)[1]
            new_image_path = self.reference_images_dir / f"{name}{ext}"
            shutil.copy2(reference_image, new_image_path)
            character_data["reference_image"] = str(new_image_path)

        self.characters[name] = character_data
        self._save_characters()

        return character_data

    def get_character(self, name: str) -> Optional[Dict]:
        """Get character data by name."""
        return self.characters.get(name)

    def list_characters(self) -> List[str]:
        """List all character names."""
        return list(self.characters.keys())

    def remove_character(self, name: str) -> bool:
        """
        Remove a character.

        Args:
            name: Character name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.characters:
            # Remove reference image if exists
            char_data = self.characters[name]
            if char_data.get("reference_image") and os.path.exists(char_data["reference_image"]):
                try:
                    os.remove(char_data["reference_image"])
                except OSError:
                    pass

            del self.characters[name]
            self._save_characters()
            return True
        return False

    def build_prompt_with_character(self,
                                    base_prompt: str,
                                    character_name: str,
                                    merge_mode: str = "prepend") -> str:
        """
        Build a prompt incorporating character description.

        Args:
            base_prompt: The original prompt
            character_name: Name of the character to use
            merge_mode: How to merge character description ("prepend", "append", "replace")

        Returns:
            Enhanced prompt with character description
        """
        character = self.get_character(character_name)
        if not character:
            print(f"Warning: Character '{character_name}' not found, using original prompt")
            return base_prompt

        char_description = character["description"]

        if merge_mode == "prepend":
            return f"{char_description}, {base_prompt}"
        elif merge_mode == "append":
            return f"{base_prompt}, {char_description}"
        elif merge_mode == "replace":
            return char_description
        else:
            return base_prompt

    def get_character_negative_prompt(self, character_name: str) -> Optional[str]:
        """Get the negative prompt for a character."""
        character = self.get_character(character_name)
        if character:
            return character.get("negative_prompt")
        return None

    def get_character_reference_image(self, character_name: str) -> Optional[str]:
        """Get the reference image path for a character."""
        character = self.get_character(character_name)
        if character:
            ref_image = character.get("reference_image")
            if ref_image and os.path.exists(ref_image):
                return ref_image
        return None

    def export_character(self, name: str, output_path: str) -> bool:
        """
        Export a character to a JSON file.

        Args:
            name: Character name
            output_path: Path to save the character JSON

        Returns:
            True if successful
        """
        character = self.get_character(name)
        if not character:
            return False

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({name: character}, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    def import_character(self, json_path: str) -> List[str]:
        """
        Import character(s) from a JSON file.

        Args:
            json_path: Path to character JSON file

        Returns:
            List of imported character names
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imported = []
            for name, char_data in data.items():
                self.characters[name] = char_data
                imported.append(name)

            self._save_characters()
            return imported
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error importing characters: {e}")
            return []


# CLI interface for character management
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage DiffuGen characters")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add character
    add_parser = subparsers.add_parser("add", help="Add a new character")
    add_parser.add_argument("name", help="Character name")
    add_parser.add_argument("description", help="Character description")
    add_parser.add_argument("--reference", help="Path to reference image")
    add_parser.add_argument("--tags", nargs="+", help="Character tags")
    add_parser.add_argument("--negative", help="Default negative prompt")

    # List characters
    subparsers.add_parser("list", help="List all characters")

    # Get character
    get_parser = subparsers.add_parser("get", help="Get character details")
    get_parser.add_argument("name", help="Character name")

    # Remove character
    remove_parser = subparsers.add_parser("remove", help="Remove a character")
    remove_parser.add_argument("name", help="Character name")

    # Export character
    export_parser = subparsers.add_parser("export", help="Export a character")
    export_parser.add_argument("name", help="Character name")
    export_parser.add_argument("output", help="Output JSON file")

    # Import character
    import_parser = subparsers.add_parser("import", help="Import character(s)")
    import_parser.add_argument("input", help="Input JSON file")

    args = parser.parse_args()

    manager = CharacterManager()

    if args.command == "add":
        char = manager.add_character(
            name=args.name,
            description=args.description,
            reference_image=args.reference,
            tags=args.tags,
            negative_prompt=args.negative
        )
        print(f"Character '{args.name}' added successfully!")
        print(json.dumps(char, indent=2))

    elif args.command == "list":
        characters = manager.list_characters()
        if characters:
            print("Available characters:")
            for char_name in characters:
                char = manager.get_character(char_name)
                print(f"  - {char_name}: {char['description'][:60]}...")
        else:
            print("No characters found.")

    elif args.command == "get":
        char = manager.get_character(args.name)
        if char:
            print(json.dumps({args.name: char}, indent=2))
        else:
            print(f"Character '{args.name}' not found.")

    elif args.command == "remove":
        if manager.remove_character(args.name):
            print(f"Character '{args.name}' removed successfully!")
        else:
            print(f"Character '{args.name}' not found.")

    elif args.command == "export":
        if manager.export_character(args.name, args.output):
            print(f"Character '{args.name}' exported to {args.output}")
        else:
            print(f"Failed to export character '{args.name}'")

    elif args.command == "import":
        imported = manager.import_character(args.input)
        if imported:
            print(f"Imported characters: {', '.join(imported)}")
        else:
            print("Failed to import characters")

    else:
        parser.print_help()
