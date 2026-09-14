from dataclasses import dataclass

@dataclass(frozen=True)
class ProjectConfig:
    label_window_days: int = 90
