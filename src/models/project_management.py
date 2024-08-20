from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid

@dataclass
class Task:
    id: str
    name: str
    description: str
    assigned_to: str
    status: str
    priority: int
    due_date: datetime
    estimated_hours: float

class ProjectManager:
    def __init__(self):
        self.tasks: List[Task] = []
        self.resources: Dict[str, Dict[str, Any]] = {}

    def create_task(self, name: str, description: str, assigned_to: str, priority: int,
                    due_date: datetime, estimated_hours: float) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            assigned_to=assigned_to,
            status="Not Started",
            priority=priority,
            due_date=due_date,
            estimated_hours=estimated_hours
        )
        self.tasks.append(task)
        return task

    def update_task_status(self, task_id: str, new_status: str) -> None:
        task = self.get_task_by_id(task_id)
        if task:
            task.status = new_status

    def get_task_by_id(self, task_id: str) -> Task:
        return next((task for task in self.tasks if task.id == task_id), None)

    def get_tasks_by_status(self, status: str) -> List[Task]:
        return [task for task in self.tasks if task.status == status]

    def get_tasks_by_assigned_to(self, assigned_to: str) -> List[Task]:
        return [task for task in self.tasks if task.assigned_to == assigned_to]

    def add_resource(self, name: str, role: str, availability: float) -> None:
        self.resources[name] = {"role": role, "availability": availability}

    def update_resource_availability(self, name: str, availability: float) -> None:
        if name in self.resources:
            self.resources[name]["availability"] = availability

    def get_resource_availability(self, name: str) -> float:
        return self.resources.get(name, {}).get("availability", 0)

    def calculate_project_progress(self) -> float:
        completed_tasks = len(self.get_tasks_by_status("Completed"))
        total_tasks = len(self.tasks)
        return (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

    def get_overdue_tasks(self) -> List[Task]:
        current_date = datetime.now()
        return [task for task in self.tasks if task.due_date < current_date and task.status != "Completed"]

    def estimate_completion_date(self) -> datetime:
        remaining_hours = sum(task.estimated_hours for task in self.tasks if task.status != "Completed")
        total_availability = sum(resource["availability"] for resource in self.resources.values())
        days_to_complete = remaining_hours / (total_availability * 8)  # Assuming 8-hour workdays
        return datetime.now() + timedelta(days=days_to_complete)

# Example usage
if __name__ == "__main__":
    pm = ProjectManager()

    # Create tasks
    pm.create_task("Implement feature X", "Develop and test feature X", "Alice", 2,
                   datetime.now() + timedelta(days=7), 20)
    pm.create_task("Refactor module Y", "Improve code quality of module Y", "Bob", 1,
                   datetime.now() + timedelta(days=5), 15)

    # Add resources
    pm.add_resource("Alice", "Developer", 0.8)
    pm.add_resource("Bob", "Developer", 1.0)

    # Update task status
    tasks = pm.get_tasks_by_assigned_to("Alice")
    if tasks:
        pm.update_task_status(tasks[0].id, "In Progress")

    # Print project progress
    print(f"Project Progress: {pm.calculate_project_progress():.2f}%")

    # Print estimated completion date
    print(f"Estimated Completion Date: {pm.estimate_completion_date()}")
