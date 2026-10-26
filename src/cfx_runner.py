#!/usr/bin/env python3
#######################################################
#                                                     #
#           Surrogate Based Optimization              #
#        Framework for the Parametric Design          #
#              of Compressor Stages                   #
#                                                     #
#######################################################
# Authors:                                            #
#    Murillo S. S. Pereira Neto,                      #
#    @ University of São Paulo, Brazil                #
#                                                     #
#                                                     #
# Description:                                        #
#   Runs one design point in ANSYS Workbench/CFX:     #
#   prepares the project, writes the journal (.wbjn), #
#   runs runwb2 in batch mode and collects outputs    #
#   and convergence.                                  #
#######################################################

#-----------------------------------------------------#
# Importing general packages
#-----------------------------------------------------#
import os
import re
import glob
import json
import time
import shutil
import subprocess
import numpy as np

#-----------------------------------------------------#
# Importing SBO packages
#-----------------------------------------------------#
from common.config import read_raw_section
from common.logger import log_print

# Workbench execution modes
#   copy_new_dp:    copy the base project, duplicate the base design point and run the new one
#   copy_modify_dp: copy the base project, modify the base design point and run it
#   base_new_dp:    add a design point to the base project itself and run the new one
#   shared_new_dp:  copy the base project once (OPT/Workbench) and add the design points of all generations to it
MODES = ("copy_new_dp", "copy_modify_dp", "base_new_dp", "shared_new_dp")


def parse_bool(raw):
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _is_none(value):
    return value is None or str(value).strip().lower() in ("none", "null", "")


class WorkbenchSettings():

    """ Settings read from the config file

    [workbench]             raw values (paths may contain spaces)
    [workbench_inputs]      CFD variable = Workbench parameter, unit
    [workbench_constants]   Workbench parameter = value, unit (fixed inputs)
    [workbench_outputs]     output column = Workbench parameter
    [cfd_convergence]       method = out_file | max_iterations
    """

    def __init__(self, cfg_filepath, IN):

        cfg_dir = os.path.dirname(cfg_filepath)
        raw = read_raw_section(cfg_filepath, "workbench")
        if "template_project" not in raw:
            raise ValueError("[workbench] template_project is missing in the config file")

        template_project = raw["template_project"]
        if not os.path.isabs(template_project):
            template_project = os.path.join(cfg_dir, template_project)
        self.template_project   = os.path.normpath(template_project)
        self.mode               = raw.get("mode", "copy_new_dp")
        self.base_design_point  = raw.get("base_design_point", "1")
        self.release            = raw.get("release", "19.2")
        self.script_version     = raw.get("script_version", "19.2.120")
        self.wb_exe             = raw.get("wb_exe", r"C:\Program Files\ANSYS Inc\v192\Framework\bin\Win64\runwb2.exe")
        self.timeout_s          = int(float(raw.get("timeout_s", 21600)))
        # Workbench is stopped if the solver does not start (no .out file) within this time (geometry/mesh failure)
        self.solver_start_timeout_s = int(float(raw.get("solver_start_timeout_s", 7200)))
        self.cfx_system         = raw.get("cfx_system", "CFX")
        self.retain_solution    = parse_bool(raw.get("retain_solution", "true"))
        self.run_workbench      = parse_bool(raw.get("run_workbench", "true"))
        self.breakpoint_before_cfd = parse_bool(raw.get("breakpoint_before_cfd", "false"))
        if self.mode not in MODES:
            raise ValueError(f"[workbench] mode = {self.mode} not in {MODES}")

        # Inputs: {cfd_var_name: (parameter, unit)}
        self.inputs = {}
        for name, val in IN.get("workbench_inputs", {}).items():
            val = val if isinstance(val, list) else [val]
            unit = val[1] if len(val) > 1 and not _is_none(val[1]) else None
            self.inputs[name] = (str(val[0]), unit)

        # Constants: {parameter: (value, unit)}
        self.constants = {}
        for parameter, val in IN.get("workbench_constants", {}).items():
            val = val if isinstance(val, list) else [val]
            unit = val[1] if len(val) > 1 and not _is_none(val[1]) else None
            self.constants[parameter] = (val[0], unit)

        # Outputs: {column_name: parameter}
        self.outputs = {}
        for name, val in IN.get("workbench_outputs", {}).items():
            self.outputs[name] = str(val[0] if isinstance(val, list) else val)

        # Convergence
        conv = IN.get("cfd_convergence", {})
        self.convergence_method = _first(conv.get("method", ["out_file"]))
        self.max_iterations     = _first(conv.get("max_iterations", [None]))
        self.iterations_output  = _first(conv.get("iterations_output", [None]))

    def validate(self, cfd_var_names, output_names):
        """ Checks that every CFD variable and required output has a valid Workbench parameter """
        errors = []
        for name in cfd_var_names:
            if name not in self.inputs:
                errors.append(f"[workbench_inputs] missing: {name}")
            elif not _is_parameter(self.inputs[name][0]):
                errors.append(f"[workbench_inputs] {name} = {self.inputs[name][0]} is not a Workbench parameter (Pxx)")
        for parameter in self.constants:
            if not _is_parameter(parameter):
                errors.append(f"[workbench_constants] {parameter} is not a Workbench parameter (Pxx)")
        for name in output_names:
            if name not in self.outputs:
                errors.append(f"[workbench_outputs] missing: {name}")
            elif not _is_parameter(self.outputs[name]):
                errors.append(f"[workbench_outputs] {name} = {self.outputs[name]} is not a Workbench parameter (Pxx)")
        if self.convergence_method == "max_iterations":
            if self.max_iterations is None or self.iterations_output is None:
                errors.append("[cfd_convergence] method = max_iterations requires max_iterations and iterations_output")
            elif self.iterations_output not in self.outputs:
                errors.append(f"[workbench_outputs] missing: {self.iterations_output} (iterations_output)")
        elif self.convergence_method not in CONVERGENCE_CHECKS:
            errors.append(f"[cfd_convergence] unknown method: {self.convergence_method}")
        if not os.path.isfile(self.template_project):
            errors.append(f"[workbench] template_project not found: {self.template_project}")
        if self.run_workbench and not os.path.isfile(self.wb_exe):
            errors.append(f"[workbench] wb_exe not found: {self.wb_exe}")
        if self.convergence_method == "out_file" and not self.retain_solution and self.mode != "copy_modify_dp":
            log_print("WARNING: retain_solution = false may delete the solver .out file needed by "
                      "[cfd_convergence] method = out_file")
        if errors:
            raise ValueError("Invalid Workbench configuration:\n  " + "\n  ".join(errors))


def _is_parameter(name):
    return re.fullmatch(r"P\d+", name) is not None


def _first(value):
    return value[0] if isinstance(value, list) else value


# -----------------------------------------------------#
# Journal
# -----------------------------------------------------#

def format_expression(value, unit=None):
    """Format value as a Workbench expression string."""
    expr = f"{float(value):.10g}" if isinstance(value, (int, float, np.number)) else str(value)
    if unit:
        expr = f"{expr} [{unit}]"
    return expr


def generate_wb_journal(journal_path, project_file, parameters, settings, results_csv, dp_name_file):
    """ Writes the .wbjn file

    parameters: {Workbench parameter: (value, unit)}
    """
    new_dp = settings.mode in ("copy_new_dp", "base_new_dp", "shared_new_dp")

    lines = [
        "# encoding: utf-8",
        f"# Release {settings.release}",
        f'SetScriptVersion(Version="{settings.script_version}")',
        f'Open(FilePath=r"{project_file}")',
        "",
        f'designPoint1 = Parameters.GetDesignPoint(Name="{settings.base_design_point}")',
    ]
    if new_dp:
        lines += [
            "designPoint2 = designPoint1.Duplicate()",
            "Parameters.SetBaseDesignPoint(DesignPoint=designPoint2)",
        ]
    else:
        lines += ["designPoint2 = designPoint1"]
    lines += [
        "",
        "# Design point name (used to find its row in the exported table)",
        f'dpNameFile = open(r"{dp_name_file}", "w")',
        "dpNameFile.write(str(designPoint2.Name))",
        "dpNameFile.close()",
        "",
    ]

    for i, (parameter, (value, unit)) in enumerate(parameters.items(), start=1):
        lines += [
            f'parameter{i} = Parameters.GetParameter(Name="{parameter}")',
            "designPoint2.SetParameterExpression(",
            f"    Parameter=parameter{i},",
            f'    Expression="{format_expression(value, unit)}")',
            "",
        ]

    lines += [
        "Save(Overwrite=True)",
        "backgroundSession1 = UpdateAllDesignPoints(DesignPoints=[designPoint2])",
        f'system1 = GetSystem(Name="{settings.cfx_system}")',
        'solution1 = system1.GetContainer(ComponentName="Solution")',
        "solution1.DisplayMonitors()",
        "Save(Overwrite=True)",
        "Parameters.SetBaseDesignPoint(DesignPoint=designPoint1)",
        "solution1.Exit()",
    ]
    if new_dp:
        lines += [f"designPoint2.Retained = {'True' if settings.retain_solution else 'False'}"]
    lines += [
        "Save(Overwrite=True)",
        "",
        "# Export all design points (parameters in/out) to CSV",
        f'Parameters.ExportAllDesignPointsData(FilePath=r"{results_csv}")',
    ]

    with open(journal_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------#
# Workbench execution
# -----------------------------------------------------#

def run_workbench(journal_path, wb_exe, timeout_s, log_dir, project_files_dir=None, solver_start_timeout_s=None):
    """ Runs runwb2 in batch mode. Returns (return code or None, message)

    Stopped after timeout_s, or after solver_start_timeout_s if no solver .out file was written in
    project_files_dir (Workbench waiting on a geometry or mesh failure)
    """
    cmd = [wb_exe, "-B", "-R", journal_path, "-X"]
    log_print(f"Running: {' '.join(cmd)}")
    t_start = time.time()
    with open(os.path.join(log_dir, "wb_stdout.txt"), "w") as fo, \
         open(os.path.join(log_dir, "wb_stderr.txt"), "w") as fe:
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, cwd=log_dir)
        message = None
        solver_started = False
        while proc.poll() is None:
            time.sleep(30)
            dt = time.time() - t_start
            if not solver_started and project_files_dir is not None:
                solver_started = len(_new_out_files(project_files_dir, t_start)) > 0
            if dt > timeout_s:
                message = f"timeout after {timeout_s} s"
            elif solver_start_timeout_s and not solver_started and dt > solver_start_timeout_s:
                message = (f"solver did not start within {solver_start_timeout_s} s "
                           f"(geometry or mesh failure?)")
            if message is not None:
                # Kill the whole process tree (runwb2 and its children)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                proc.wait()
                return None, message
        return_code = proc.returncode
    if return_code != 0:
        return return_code, f"Workbench exited with code {return_code} (see wb_stderr.txt)"
    return return_code, "Workbench run completed"


def _new_out_files(project_files_dir, t_start):
    return [f for f in glob.glob(os.path.join(project_files_dir, "**", "*.out"), recursive=True)
            if os.path.getmtime(f) >= t_start - 1]


def workbench_is_running():
    """ True if a Workbench process (runwb2, AnsysFW*) is running """
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return ("runwb2" in out) or ("ansysfw" in out)


def remove_stale_lock(project_file):
    """ A Workbench killed by a timeout leaves <project>_files/.lock and the next run cannot open the project.
    The lock is removed if no Workbench is running; otherwise an exception is raised (project in use) """
    lock_file = os.path.join(os.path.splitext(project_file)[0] + "_files", ".lock")
    if not os.path.isfile(lock_file):
        return
    if workbench_is_running():
        raise Exception(f"Workbench project in use ({lock_file}). Close Workbench and run again.")
    log_print(f"WARNING: removing stale lock of the Workbench project: {lock_file}")
    os.remove(lock_file)


# -----------------------------------------------------#
# Results
# -----------------------------------------------------#

def read_design_point_table(csv_path):
    """ Reads the table exported by Parameters.ExportAllDesignPointsData

    Comment lines start with '#'; the first non-comment line is the header
    ("Name, P1, P2, ..."). Header entries such as "P12 - Name [unit]" are reduced to "P12".
    Returns a list of {column: raw string}.
    """
    rows, header = [], None
    # utf-8-sig: Workbench writes a byte order mark at the start of the file
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip().strip('"') for field in line.split(",")]
            if header is None:
                header = []
                for field in fields:
                    match = re.match(r"^(P\d+)\b", field)
                    header.append(match.group(1) if match else field)
                continue
            rows.append(dict(zip(header, fields)))
    return rows


def select_design_point_row(rows, dp_name):
    """ Row of design point `dp_name` (falls back to the last row) """
    if not rows:
        return None, "empty design point table"
    name_column = list(rows[0].keys())[0]
    candidates = set()
    if dp_name is not None:
        candidates = {dp_name, f"DP {dp_name}", f"DP{dp_name}"}
        if dp_name == "0":
            candidates.add("Current")
    for row in rows:
        if row[name_column] in candidates:
            return row, f"design point '{row[name_column]}'"
    return rows[-1], f"design point '{dp_name}' not found in table, using last row '{rows[-1][name_column]}'"


def _to_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


# -----------------------------------------------------#
# Convergence checks: (settings, run_dir, project_files_dir, t_start, outputs) -> (0/1, message)
# -----------------------------------------------------#

def check_convergence_out_file(settings, run_dir, project_files_dir, t_start, outputs):
    """ Converged if the solver stopped by an interrupt condition or residual target
    (parameter stabilization), not converged if it stopped at the maximum number of
    iterations or did not finish """
    out_files = _new_out_files(project_files_dir, t_start)
    if not out_files:
        return 0, "no solver .out file written during this run"
    out_file = max(out_files, key=os.path.getmtime)
    with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    title = "Termination and Interrupt Condition Summary"
    index = text.rfind(title)
    if index < 0:
        return 0, f"solver did not finish (no termination summary in {out_file})"
    # Block between the summary title and the next section
    block_lines = []
    for line in text[index + len(title):].splitlines()[1:]:
        if line.strip().startswith("====") and block_lines:
            break
        if line.strip() and not line.strip().startswith("===="):
            block_lines.append(line.strip())
    block = " | ".join(block_lines)
    lower = block.lower()
    if "maximum number of" in lower:
        return 0, f"maximum number of iterations reached ({block})"
    if "interrupt condition" in lower or "target" in lower:
        return 1, f"solver stopped by: {block}"
    return 0, f"unrecognized termination condition: {block}"


def check_convergence_max_iterations(settings, run_dir, project_files_dir, t_start, outputs):
    """ Converged if the number of iterations is below the maximum """
    iterations = outputs.get(settings.iterations_output, np.nan)
    if np.isnan(iterations):
        return 0, f"{settings.iterations_output} not available"
    if iterations < float(settings.max_iterations):
        return 1, f"{settings.iterations_output} = {iterations:g} < {settings.max_iterations}"
    return 0, f"{settings.iterations_output} = {iterations:g} >= {settings.max_iterations}"


CONVERGENCE_CHECKS = {
    "out_file": check_convergence_out_file,
    "max_iterations": check_convergence_max_iterations,
}


# -----------------------------------------------------#
# Runner
# -----------------------------------------------------#

class CFXRunner():

    """ Runs one design point inside `run_dir`

    Files in run_dir:
        <project>.wbpj, <project>_files     copy of the base project (copy_* modes; shared_new_dp: in
                                            shared_project_dir, copied only once)
        run.wbjn                            journal
        run_info.json                       start time and project used (restart)
        dp_name.txt                         name of the design point that was run
        results.csv                         design point table exported by Workbench
        wb_stdout.txt, wb_stderr.txt        Workbench log
    """

    def __init__(self, settings, shared_project_dir=None):
        self.settings = settings
        self.shared_project_dir = shared_project_dir

    def _project_file(self, run_dir):
        if self.settings.mode == "base_new_dp":
            return self.settings.template_project
        if self.settings.mode == "shared_new_dp":
            return os.path.join(self.shared_project_dir, os.path.basename(self.settings.template_project))
        return os.path.join(run_dir, os.path.basename(self.settings.template_project))

    def _copy_project(self, project_file):
        files_dir = self._files_dir(project_file)
        if os.path.exists(files_dir):
            shutil.rmtree(files_dir)
        log_print(f"Copying base project to {os.path.dirname(project_file)}")
        os.makedirs(os.path.dirname(project_file), exist_ok=True)
        shutil.copy2(self.settings.template_project, project_file)
        template_files_dir = self._files_dir(self.settings.template_project)
        if os.path.isdir(template_files_dir):
            shutil.copytree(template_files_dir, files_dir)

    @staticmethod
    def _files_dir(project_file):
        return os.path.splitext(project_file)[0] + "_files"

    def prepare(self, run_dir, cfd_values):
        """ Copies the project (if needed) and writes the journal. Returns the journal path """
        os.makedirs(run_dir, exist_ok=True)
        project_file = self._project_file(run_dir)
        if self.settings.mode in ("copy_new_dp", "copy_modify_dp"):
            self._copy_project(project_file)
        elif self.settings.mode == "shared_new_dp" and not os.path.isfile(project_file):
            self._copy_project(project_file)
        elif self.settings.mode == "shared_new_dp":
            log_print(f"Adding design point to {project_file}")
        remove_stale_lock(project_file)

        parameters = {}
        for name, value in cfd_values.items():
            parameter, unit = self.settings.inputs[name]
            parameters[parameter] = (value, unit)
        for parameter, (value, unit) in self.settings.constants.items():
            parameters[parameter] = (value, unit)

        journal_path = os.path.join(run_dir, "run.wbjn")
        for filename in ("results.csv", "dp_name.txt"):
            if os.path.exists(os.path.join(run_dir, filename)):
                os.remove(os.path.join(run_dir, filename))
        generate_wb_journal(journal_path, project_file, parameters, self.settings,
                            results_csv=os.path.join(run_dir, "results.csv"),
                            dp_name_file=os.path.join(run_dir, "dp_name.txt"))
        log_print(f"Journal written: {journal_path}")
        return journal_path

    def execute(self, run_dir, journal_path):
        """ Runs Workbench and records the run information """
        t_start = time.time()
        info = {"t_start": t_start, "project_file": self._project_file(run_dir), "mode": self.settings.mode}
        _write_json(os.path.join(run_dir, "run_info.json"), info)
        return_code, message = run_workbench(journal_path, self.settings.wb_exe, self.settings.timeout_s, run_dir,
                                             project_files_dir=self._files_dir(info["project_file"]),
                                             solver_start_timeout_s=self.settings.solver_start_timeout_s)
        info.update({"t_end": time.time(), "return_code": return_code, "message": message})
        _write_json(os.path.join(run_dir, "run_info.json"), info)
        log_print(message)
        return info

    def collect(self, run_dir):
        """ Reads outputs and convergence of the last run in run_dir

        Returns {"outputs": {column: value}, "Convergence": 0/1, "message": str}
        """
        info = _read_json(os.path.join(run_dir, "run_info.json")) or {}
        t_start = info.get("t_start", 0.0)
        project_file = info.get("project_file", self._project_file(run_dir))
        messages = [info.get("message", "")]

        outputs = {name: np.nan for name in self.settings.outputs}
        results_csv = os.path.join(run_dir, "results.csv")
        if os.path.isfile(results_csv):
            dp_name = None
            dp_name_file = os.path.join(run_dir, "dp_name.txt")
            if os.path.isfile(dp_name_file):
                with open(dp_name_file, "r") as f:
                    dp_name = f.read().strip()
            row, message = select_design_point_row(read_design_point_table(results_csv), dp_name)
            messages.append(message)
            if row is not None:
                for name, parameter in self.settings.outputs.items():
                    outputs[name] = _to_float(row.get(parameter))
        else:
            messages.append("results.csv not found")

        convergence, message = CONVERGENCE_CHECKS[self.settings.convergence_method](
            self.settings, run_dir, self._files_dir(project_file), t_start, outputs)
        messages.append(message)
        # Workbench failure or timeout: the convergence of the solver (.out) is kept, since the solver may have
        # converged before the failure (e.g. timeout during post processing); the outputs are nan in that case
        if info.get("return_code") != 0 and convergence == 1:
            messages.append("Workbench failed after the solver converged: outputs not available")
        # The journal did not create the design point (e.g. project could not be opened): convergence unknown
        if not os.path.isfile(os.path.join(run_dir, "dp_name.txt")):
            messages.append("design point not created: convergence unknown")
            return {"outputs": outputs, "Convergence": np.nan, "message": "; ".join(m for m in messages if m)}
        return {"outputs": outputs, "Convergence": int(convergence),
                "message": "; ".join(m for m in messages if m)}

    def run(self, run_dir, cfd_values):
        journal_path = self.prepare(run_dir, cfd_values)
        self.execute(run_dir, journal_path)
        return self.collect(run_dir)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=float)


def _read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
