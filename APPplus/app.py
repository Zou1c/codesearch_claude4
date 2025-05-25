#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeSearch - 智能代码搜索和管理系统后端
支持项目管理、语义搜索、RAG、GitHub集成等功能
"""

# pip install flask flask-cors numpy scikit-learn sentence-transformers GitPython

import os
import sys
import json
import ast
import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Web框架和CORS
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 向量搜索和嵌入
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Git支持
import git
from git import Repo

# 异步任务
import threading
import queue

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FunctionInfo:
    """函数信息类"""
    name: str
    file_path: str
    line_number: int
    doc_string: str
    parameters: List[str]
    return_type: str
    complexity: int
    dependencies: List[str]
    source_code: str
    file_hash: str

@dataclass  
class FileInfo:
    """文件信息类"""
    path: str
    name: str
    extension: str
    size: int
    lines: int
    functions: List[FunctionInfo]
    imports: List[str]
    classes: List[str]
    hash: str
    last_modified: str

@dataclass
class ProjectInfo:
    """项目信息类"""
    name: str
    path: str
    files: List[FileInfo]
    total_files: int
    total_lines: int
    languages: List[str]
    dependencies: Dict[str, List[str]]
    created_at: str
    updated_at: str
    git_url: Optional[str] = None
    git_branch: Optional[str] = None

class CodeAnalyzer:
    """代码分析器"""
    
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs', '.php'}
    
    def __init__(self):
        self.function_parsers = {
            '.py': self._parse_python_functions,
            '.js': self._parse_javascript_functions,
            '.ts': self._parse_typescript_functions,
            '.java': self._parse_java_functions,
        }
    
    def analyze_file(self, file_path: str) -> FileInfo:
        """分析单个文件"""
        try:
            path_obj = Path(file_path)
            extension = path_obj.suffix.lower()
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_hash = hashlib.md5(content.encode()).hexdigest()
            lines = len(content.split('\n'))
            
            functions = []
            imports = []
            classes = []
            
            if extension in self.function_parsers:
                functions = self.function_parsers[extension](content, file_path)
            
            if extension == '.py':
                imports, classes = self._extract_python_metadata(content)
            
            return FileInfo(
                path=file_path,
                name=path_obj.name,
                extension=extension,
                size=len(content),
                lines=lines,
                functions=functions,
                imports=imports,
                classes=classes,
                hash=file_hash,
                last_modified=datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
            )
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {e}")
            return None
    
    def _parse_python_functions(self, content: str, file_path: str) -> List[FunctionInfo]:
        """解析Python函数"""
        functions = []
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = self._extract_python_function_info(node, content, file_path)
                    functions.append(func_info)
        except Exception as e:
            logger.error(f"解析Python文件 {file_path} 时出错: {e}")
        
        return functions
    
    def _extract_python_function_info(self, node: ast.FunctionDef, content: str, file_path: str) -> FunctionInfo:
        """提取Python函数详细信息"""
        lines = content.split('\n')
        
        # 获取函数源码
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
        source_lines = lines[start_line:min(end_line, len(lines))]
        source_code = '\n'.join(source_lines)
        
        # 获取文档字符串
        doc_string = ""
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            doc_string = node.body[0].value.value
        
        # 获取参数
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            if arg.annotation:
                param_name += f": {ast.unparse(arg.annotation)}"
            parameters.append(param_name)
        
        # 获取返回类型
        return_type = ""
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # 计算复杂度（简单的行数计算）
        complexity = len(source_lines)
        
        # 获取依赖（简单的变量引用）
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.append(child.id)
        
        dependencies = list(set(dependencies))[:10]  # 限制依赖数量
        
        return FunctionInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            doc_string=doc_string,
            parameters=parameters,
            return_type=return_type,
            complexity=complexity,
            dependencies=dependencies,
            source_code=source_code,
            file_hash=hashlib.md5(source_code.encode()).hexdigest()
        )
    
    def _parse_javascript_functions(self, content: str, file_path: str) -> List[FunctionInfo]:
        """解析JavaScript函数（简化版）"""
        functions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if ('function ' in line and '(' in line) or ('const ' in line and '=>' in line):
                # 简化的函数检测
                func_name = self._extract_js_function_name(line)
                if func_name:
                    functions.append(FunctionInfo(
                        name=func_name,
                        file_path=file_path,
                        line_number=i + 1,
                        doc_string="",
                        parameters=[],
                        return_type="",
                        complexity=1,
                        dependencies=[],
                        source_code=line,
                        file_hash=hashlib.md5(line.encode()).hexdigest()
                    ))
        
        return functions
    
    def _parse_typescript_functions(self, content: str, file_path: str) -> List[FunctionInfo]:
        """解析TypeScript函数（简化版）"""
        return self._parse_javascript_functions(content, file_path)
    
    def _parse_java_functions(self, content: str, file_path: str) -> List[FunctionInfo]:
        """解析Java函数（简化版）"""
        functions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if ('public ' in line or 'private ' in line or 'protected ' in line) and '(' in line and ')' in line:
                func_name = self._extract_java_function_name(line)
                if func_name:
                    functions.append(FunctionInfo(
                        name=func_name,
                        file_path=file_path,
                        line_number=i + 1,
                        doc_string="",
                        parameters=[],
                        return_type="",
                        complexity=1,
                        dependencies=[],
                        source_code=line,
                        file_hash=hashlib.md5(line.encode()).hexdigest()
                    ))
        
        return functions
    
    def _extract_js_function_name(self, line: str) -> Optional[str]:
        """提取JavaScript函数名"""
        try:
            if 'function ' in line:
                parts = line.split('function')[1].split('(')[0].strip()
                return parts if parts else None
            elif 'const ' in line and '=>' in line:
                parts = line.split('const')[1].split('=')[0].strip()
                return parts if parts else None
        except:
            return None
        return None
    
    def _extract_java_function_name(self, line: str) -> Optional[str]:
        """提取Java函数名"""
        try:
            parts = line.split('(')[0].split()
            return parts[-1] if parts else None
        except:
            return None
        return None
    
    def _extract_python_metadata(self, content: str) -> Tuple[List[str], List[str]]:
        """提取Python文件的导入和类信息"""
        imports = []
        classes = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except:
            pass
        
        return imports, classes

class VectorSearchEngine:
    """向量搜索引擎 - 支持RAG"""
    
    def __init__(self):
        self.model = None
        self.function_embeddings = {}
        self.function_data = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.initialized = False
        
    def initialize(self):
        """初始化embedding模型"""
        try:
            logger.info("正在初始化Sentence Transformer模型...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.initialized = True
            logger.info("模型初始化完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            logger.info("将使用TF-IDF作为备选方案")
    
    def add_functions(self, functions: List[FunctionInfo]):
        """添加函数到搜索索引"""
        if not functions:
            return
        
        # 准备文本数据
        texts = []
        for func in functions:
            text = f"{func.name} {func.doc_string} {' '.join(func.parameters)} {func.source_code}"
            texts.append(text)
            key = f"{func.file_path}:{func.line_number}"
            self.function_data[key] = func
        
        # 如果有embedding模型，生成向量
        if self.initialized and self.model:
            try:
                embeddings = self.model.encode(texts)
                for i, func in enumerate(functions):
                    key = f"{func.file_path}:{func.line_number}"
                    self.function_embeddings[key] = embeddings[i]
            except Exception as e:
                logger.error(f"生成embedding时出错: {e}")
        
        # 更新TF-IDF矩阵
        try:
            all_texts = [f"{func.name} {func.doc_string} {func.source_code}" 
                        for func in self.function_data.values()]
            if all_texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        except Exception as e:
            logger.error(f"更新TF-IDF矩阵时出错: {e}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[FunctionInfo, float]]:
        """搜索相关函数"""
        if not self.function_data:
            return []
        
        results = []
        
        # 优先使用embedding搜索
        if self.initialized and self.model and self.function_embeddings:
            try:
                query_embedding = self.model.encode([query])
                similarities = []
                
                for key, func_embedding in self.function_embeddings.items():
                    similarity = cosine_similarity(query_embedding.reshape(1, -1), 
                                                 func_embedding.reshape(1, -1))[0][0]
                    similarities.append((key, similarity))
                
                # 排序并获取top结果
                similarities.sort(key=lambda x: x[1], reverse=True)
                for key, score in similarities[:top_k]:
                    results.append((self.function_data[key], float(score)))
                
                return results
            except Exception as e:
                logger.error(f"Embedding搜索失败: {e}")
        
        # 备选TF-IDF搜索
        if self.tfidf_matrix is not None:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
                
                # 获取排序后的索引
                indices = similarities.argsort()[::-1][:top_k]
                
                function_list = list(self.function_data.values())
                for idx in indices:
                    if idx < len(function_list) and similarities[idx] > 0:
                        results.append((function_list[idx], float(similarities[idx])))
                
                return results
            except Exception as e:
                logger.error(f"TF-IDF搜索失败: {e}")
        
        # 最后的关键词匹配
        query_lower = query.lower()
        for func in self.function_data.values():
            score = 0
            if query_lower in func.name.lower():
                score += 1.0
            if query_lower in func.doc_string.lower():
                score += 0.5
            if query_lower in func.source_code.lower():
                score += 0.3
            
            if score > 0:
                results.append((func, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class GitHubIntegration:
    """GitHub集成"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.repos = {}
    
    def clone_repository(self, repo_url: str, project_name: str, branch: str = 'main') -> str:
        """克隆GitHub仓库"""
        try:
            repo_path = self.storage_path / 'repos' / project_name
            
            # 如果目录已存在，先删除
            if repo_path.exists():
                shutil.rmtree(repo_path)
            
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"正在克隆仓库: {repo_url}")
            repo = Repo.clone_from(repo_url, repo_path, branch=branch)
            
            self.repos[project_name] = {
                'repo': repo,
                'url': repo_url,
                'branch': branch,
                'path': str(repo_path)
            }
            
            logger.info(f"仓库克隆完成: {repo_path}")
            return str(repo_path)
        except Exception as e:
            logger.error(f"克隆仓库失败: {e}")
            raise e
    
    def pull_updates(self, project_name: str) -> bool:
        """拉取更新"""
        try:
            if project_name not in self.repos:
                return False
            
            repo = self.repos[project_name]['repo']
            origin = repo.remotes.origin
            origin.pull()
            
            logger.info(f"项目 {project_name} 更新完成")
            return True
        except Exception as e:
            logger.error(f"拉取更新失败: {e}")
            return False

class ProjectManager:
    """项目管理器"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.projects = {}
        self.analyzer = CodeAnalyzer()
        self.search_engine = VectorSearchEngine()
        self.github = GitHubIntegration(storage_path)
        
        # 异步初始化搜索引擎
        threading.Thread(target=self.search_engine.initialize, daemon=True).start()
        
        self._load_projects()
    
    def _load_projects(self):
        """加载已有项目"""
        try:
            projects_file = self.storage_path / 'projects.json'
            if projects_file.exists():
                with open(projects_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for project_data in data:
                        project = ProjectInfo(**project_data)
                        self.projects[project.name] = project
                        
                        # 重建搜索索引
                        all_functions = []
                        for file_info in project.files:
                            all_functions.extend(file_info.functions)
                        self.search_engine.add_functions(all_functions)
        except Exception as e:
            logger.error(f"加载项目失败: {e}")
    
    def _save_projects(self):
        """保存项目信息"""
        try:
            projects_file = self.storage_path / 'projects.json'
            data = [asdict(project) for project in self.projects.values()]
            with open(projects_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存项目失败: {e}")
    
    def create_project(self, name: str, source_path: str = None, git_url: str = None) -> ProjectInfo:
        """创建新项目"""
        if name in self.projects:
            raise ValueError(f"项目 {name} 已存在")
        
        project_path = self.storage_path / name
        project_path.mkdir(parents=True, exist_ok=True)
        
        # 如果是Git仓库
        if git_url:
            source_path = self.github.clone_repository(git_url, name)
        
        # 如果提供了源路径，复制文件
        if source_path and Path(source_path).exists():
            self._copy_project_files(source_path, project_path)
        
        # 分析项目
        project = self._analyze_project(name, str(project_path), git_url)
        self.projects[name] = project
        self._save_projects()
        
        return project
    
    def _copy_project_files(self, source: str, dest: Path):
        """复制项目文件"""
        source_path = Path(source)
        
        if source_path.is_file():
            shutil.copy2(source_path, dest / source_path.name)
        else:
            for item in source_path.rglob('*'):
                if item.is_file() and not self._should_ignore_file(item):
                    rel_path = item.relative_to(source_path)
                    dest_file = dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_file)
    
    def _should_ignore_file(self, path: Path) -> bool:
        """判断是否应该忽略文件"""
        ignore_patterns = {
            '.git', '.gitignore', '__pycache__', '.pytest_cache',
            'node_modules', '.vscode', '.idea', '*.pyc', '*.pyo',
            '.DS_Store', 'Thumbs.db'
        }
        
        return any(pattern in str(path) for pattern in ignore_patterns)
    
    def _analyze_project(self, name: str, project_path: str, git_url: str = None) -> ProjectInfo:
        """分析项目"""
        path_obj = Path(project_path)
        files = []
        all_functions = []
        languages = set()
        total_lines = 0
        
        # 分析所有代码文件
        for file_path in path_obj.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in CodeAnalyzer.SUPPORTED_EXTENSIONS and
                not self._should_ignore_file(file_path)):
                
                file_info = self.analyzer.analyze_file(str(file_path))
                if file_info:
                    files.append(file_info)
                    all_functions.extend(file_info.functions)
                    languages.add(file_info.extension)
                    total_lines += file_info.lines
        
        # 添加到搜索引擎
        if all_functions:
            self.search_engine.add_functions(all_functions)
        
        # 构建依赖关系
        dependencies = self._build_dependencies(files)
        
        project = ProjectInfo(
            name=name,
            path=project_path,
            files=files,
            total_files=len(files),
            total_lines=total_lines,
            languages=list(languages),
            dependencies=dependencies,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            git_url=git_url
        )
        
        return project
    
    def _build_dependencies(self, files: List[FileInfo]) -> Dict[str, List[str]]:
        """构建文件依赖关系"""
        dependencies = {}
        
        for file_info in files:
            deps = []
            for import_name in file_info.imports:
                # 简单的依赖查找
                for other_file in files:
                    if (import_name in other_file.name or 
                        any(import_name in class_name for class_name in other_file.classes)):
                        deps.append(other_file.path)
            
            dependencies[file_info.path] = deps
        
        return dependencies
    
    def upload_files(self, project_name: str, files) -> bool:
        """上传文件到项目"""
        try:
            if project_name not in self.projects:
                return False
            
            project_path = Path(self.projects[project_name].path)
            
            for file in files:
                filename = secure_filename(file.filename)
                file_path = project_path / filename
                file.save(str(file_path))
            
            # 重新分析项目
            self.projects[project_name] = self._analyze_project(
                project_name, 
                str(project_path),
                self.projects[project_name].git_url
            )
            self._save_projects()
            
            return True
        except Exception as e:
            logger.error(f"上传文件失败: {e}")
            return False
    
    def upload_directory(self, project_name: str, zip_file) -> bool:
        """上传目录（zip文件）"""
        try:
            if project_name not in self.projects:
                return False
            
            project_path = Path(self.projects[project_name].path)
            
            # 创建临时目录
            temp_path = project_path / 'temp_upload'
            temp_path.mkdir(exist_ok=True)
            
            # 保存zip文件
            zip_path = temp_path / 'upload.zip'
            zip_file.save(str(zip_path))
            
            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # 移动文件到项目目录
            for item in temp_path.rglob('*'):
                if item.is_file() and item.name != 'upload.zip':
                    rel_path = item.relative_to(temp_path)
                    dest_path = project_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dest_path))
            
            # 清理临时目录
            shutil.rmtree(temp_path)
            
            # 重新分析项目
            self.projects[project_name] = self._analyze_project(
                project_name,
                str(project_path),
                self.projects[project_name].git_url
            )
            self._save_projects()
            
            return True
        except Exception as e:
            logger.error(f"上传目录失败: {e}")
            return False
    
    def search_functions(self, query: str, project_name: str = None, top_k: int = 10):
        """搜索函数"""
        return self.search_engine.search(query, top_k)
    
    def get_file_content(self, project_name: str, file_path: str) -> Optional[str]:
        """获取文件内容"""
        try:
            if project_name not in self.projects:
                return None
            
            full_path = Path(file_path)
            if not full_path.exists():
                # 尝试相对路径
                project_path = Path(self.projects[project_name].path)
                full_path = project_path / file_path
            
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"读取文件内容失败: {e}")
        
        return None
    
    def update_file_content(self, project_name: str, file_path: str, content: str) -> bool:
        """更新文件内容"""
        try:
            if project_name not in self.projects:
                return False
            
            full_path = Path(file_path)
            if not full_path.exists():
                project_path = Path(self.projects[project_name].path)
                full_path = project_path / file_path
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 重新分析文件
            file_info = self.analyzer.analyze_file(str(full_path))
            if file_info:
                # 更新项目中的文件信息
                project = self.projects[project_name]
                for i, existing_file in enumerate(project.files):
                    if existing_file.path == str(full_path):
                        project.files[i] = file_info
                        break
                
                # 更新搜索索引
                self.search_engine.add_functions(file_info.functions)
                self._save_projects()
            
            return True
        except Exception as e:
            logger.error(f"更新文件内容失败: {e}")
            return False

# Flask应用
app = Flask(__name__)
CORS(app)

# 初始化项目管理器
STORAGE_PATH = './codesearch_data'
project_manager = ProjectManager(STORAGE_PATH)

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """获取所有项目"""
    projects = [asdict(project) for project in project_manager.projects.values()]
    return jsonify({'projects': projects})

@app.route('/api/projects', methods=['POST'])
def create_project():
    """创建新项目"""
    try:
        data = request.get_json()
        name = data.get('name')
        git_url = data.get('git_url')
        
        if not name:
            return jsonify({'error': '项目名称不能为空'}), 400
        
        project = project_manager.create_project(name, git_url=git_url)
        return jsonify({'project': asdict(project)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/upload', methods=['POST'])
def upload_files(project_name):
    """上传文件"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': '没有文件'}), 400
        
        files = request.files.getlist('files')
        success = project_manager.upload_files(project_name, files)
        
        if success:
            return jsonify({'message': '上传成功'})
        else:
            return jsonify({'error': '上传失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/upload-directory', methods=['POST'])
def upload_directory(project_name):
    """上传目录"""
    try:
        if 'directory' not in request.files:
            return jsonify({'error': '没有目录文件'}), 400
        
        directory_file = request.files['directory']
        success = project_manager.upload_directory(project_name, directory_file)
        
        if success:
            return jsonify({'message': '目录上传成功'})
        else:
            return jsonify({'error': '目录上传失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_functions():
    """搜索函数"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        project_name = data.get('project_name')
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': '搜索关键词不能为空'}), 400
        
        results = project_manager.search_functions(query, project_name, top_k)
        
        # 转换结果格式
        formatted_results = []
        for func_info, score in results:
            result = asdict(func_info)
            result['score'] = score
            formatted_results.append(result)
        
        return jsonify({'results': formatted_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/files/<path:file_path>', methods=['GET'])
def get_file_content(project_name, file_path):
    """获取文件内容"""
    try:
        content = project_manager.get_file_content(project_name, file_path)
        if content is not None:
            return jsonify({'content': content})
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/files/<path:file_path>', methods=['PUT'])
def update_file_content(project_name, file_path):
    """更新文件内容"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        
        success = project_manager.update_file_content(project_name, file_path, content)
        if success:
            return jsonify({'message': '文件更新成功'})
        else:
            return jsonify({'error': '文件更新失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>', methods=['GET'])
def get_project_details(project_name):
    """获取项目详情"""
    try:
        if project_name not in project_manager.projects:
            return jsonify({'error': '项目不存在'}), 404
        
        project = project_manager.projects[project_name]
        return jsonify({'project': asdict(project)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/git/pull', methods=['POST'])
def pull_git_updates(project_name):
    """拉取Git更新"""
    try:
        success = project_manager.github.pull_updates(project_name)
        if success:
            # 重新分析项目
            project = project_manager.projects[project_name]
            updated_project = project_manager._analyze_project(
                project_name, 
                project.path, 
                project.git_url
            )
            project_manager.projects[project_name] = updated_project
            project_manager._save_projects()
            
            return jsonify({'message': 'Git更新成功'})
        else:
            return jsonify({'error': 'Git更新失败'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project_name>/dependencies', methods=['GET'])
def get_project_dependencies(project_name):
    """获取项目依赖关系"""
    try:
        if project_name not in project_manager.projects:
            return jsonify({'error': '项目不存在'}), 404
        
        project = project_manager.projects[project_name]
        return jsonify({'dependencies': project.dependencies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'search_engine_initialized': project_manager.search_engine.initialized,
        'projects_count': len(project_manager.projects),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("启动CodeSearch后端服务...")
    logger.info(f"数据存储路径: {STORAGE_PATH}")
    logger.info("服务地址: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
