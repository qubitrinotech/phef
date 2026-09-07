#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${SOURCE_DIR:-$HOME/agi}"
MOUNT_POINT="${MOUNT_POINT:-/data}"
BACKUP_ROOT="${BACKUP_ROOT:-$MOUNT_POINT/backups/agi}"
KEEP_SNAPSHOTS="${KEEP_SNAPSHOTS:-3}"
LOCK_FILE="${LOCK_FILE:-/tmp/agi-backup.lock}"
DRY_RUN=false

usage() {
    cat <<'EOF'
Использование:
  bash scripts/backup_agi_to_data.sh [--dry-run]

Создаёт новый полностью независимый backup-снимок ~/agi (проект PHEF)
в /data/backups/agi. После успешного создания хранит только 3 последних снимка:
четвёртый запуск вытесняет самый старый.
Снимки не используют hard links, символьные ссылки или --link-dest.

Имена снимков: YYYY-MM-DD_HH-MM-SS

Включаются: код, docs, phef/results (веса .pt), phef/logs, phef/output,
requirements-*.txt, python-version.txt и вся история .git.

Не включаются: .venv, кэши mypy/pytest/ruff, __pycache__, bytecode,
coverage, временные XML-отчёты и .pid-файлы мёртвых процессов.

Переменные окружения (при необходимости):
  SOURCE_DIR=/путь/к/agi
  MOUNT_POINT=/точка/монтирования
  BACKUP_ROOT=/путь/к/каталогу/снимков
  KEEP_SNAPSHOTS=3
EOF
}

case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Ошибка: неизвестный параметр: $1" >&2; usage >&2; exit 2 ;;
esac

command -v rsync >/dev/null || { echo "Ошибка: rsync не установлен." >&2; exit 1; }
command -v findmnt >/dev/null || { echo "Ошибка: findmnt не установлен." >&2; exit 1; }
command -v flock >/dev/null || { echo "Ошибка: flock не установлен." >&2; exit 1; }

[[ "$KEEP_SNAPSHOTS" =~ ^[1-9][0-9]*$ ]] || {
    echo "Ошибка: KEEP_SNAPSHOTS должен быть положительным целым числом." >&2
    exit 1
}

SOURCE_DIR="$(realpath -e "$SOURCE_DIR")"
MOUNT_POINT="$(realpath -e "$MOUNT_POINT")"
BACKUP_ROOT="$(realpath -m "$BACKUP_ROOT")"

[[ -d "$SOURCE_DIR" ]] || { echo "Ошибка: каталог проекта не найден: $SOURCE_DIR" >&2; exit 1; }
[[ -f "$SOURCE_DIR/pyproject.toml" ]] || {
    echo "Ошибка: $SOURCE_DIR не похож на корень AGI/PHEF: отсутствует pyproject.toml." >&2
    exit 1
}
[[ -d "$SOURCE_DIR/phef" ]] || {
    echo "Ошибка: $SOURCE_DIR не похож на корень AGI/PHEF: отсутствует каталог phef/." >&2
    exit 1
}
[[ "$SOURCE_DIR" != "$MOUNT_POINT" && "$SOURCE_DIR" != "$MOUNT_POINT"/* ]] || {
    echo "Ошибка: исходный каталог не должен находиться внутри backup-диска: $SOURCE_DIR" >&2
    exit 1
}
[[ "$BACKUP_ROOT" == "$MOUNT_POINT"/* ]] || {
    echo "Ошибка: BACKUP_ROOT должен находиться внутри $MOUNT_POINT: $BACKUP_ROOT" >&2
    exit 1
}

MOUNT_SOURCE="$(findmnt -rn -o SOURCE --target "$MOUNT_POINT" || true)"
MOUNT_TARGET="$(findmnt -rn -o TARGET --target "$MOUNT_POINT" || true)"
[[ -n "$MOUNT_SOURCE" && "$MOUNT_TARGET" == "$MOUNT_POINT" ]] || {
    echo "Ошибка: $MOUNT_POINT не является отдельной активной точкой монтирования." >&2
    exit 1
}

mkdir -p "$BACKUP_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Ошибка: уже выполняется другой backup AGI (lock: $LOCK_FILE)." >&2; exit 1; }

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
SNAPSHOT_DIR="$BACKUP_ROOT/$TIMESTAMP"
SUFFIX=0
while [[ -e "$SNAPSHOT_DIR" ]]; do
    SUFFIX=$((SUFFIX + 1))
    SNAPSHOT_DIR="$BACKUP_ROOT/${TIMESTAMP}_$(printf '%02d' "$SUFFIX")"
done

RSYNC_ARGS=(
    --archive
    --human-readable
    --itemize-changes
    --partial
    --info=stats2,progress2
    --exclude=/.venv/
    --exclude=/.mypy_cache/
    --exclude=/.pytest_cache/
    --exclude=/.ruff_cache/
    --exclude=/.coverage
    --exclude=/htmlcov/
    --exclude=/report.xml
    --exclude=/coverage.xml
    --exclude='__pycache__/'
    --exclude='*.py[cod]'
    --exclude='*.so'
    --exclude='*.egg-info/'
    --exclude='*.pid'
    --exclude='.DS_Store'
)

if "$DRY_RUN"; then
    RSYNC_ARGS+=(--dry-run)
fi

SOURCE_SIZE="$(du -sh --exclude=.venv --exclude=__pycache__ "$SOURCE_DIR" 2>/dev/null | cut -f1 || echo '?')"
FREE_SPACE="$(df -h --output=avail "$MOUNT_POINT" 2>/dev/null | tail -1 | tr -d ' ' || echo '?')"

printf 'Источник:        %s/ (%s)\n' "$SOURCE_DIR" "$SOURCE_SIZE"
printf 'Каталог снимков: %s/\n' "$BACKUP_ROOT"
printf 'Новый снимок:    %s/\n' "$SNAPSHOT_DIR"
printf 'Диск:            %s (%s), свободно %s\n' "$MOUNT_POINT" "$MOUNT_SOURCE" "$FREE_SPACE"
printf 'Хранить:         %s последних снимка(ов)\n' "$KEEP_SNAPSHOTS"
printf 'Режим:           %s\n\n' "$([ "$DRY_RUN" = true ] && echo 'проверка без изменений' || echo 'создание нового полного снимка')"

if "$DRY_RUN"; then
    rsync "${RSYNC_ARGS[@]}" "$SOURCE_DIR/" "$SNAPSHOT_DIR/"
    printf '\nВ режиме --dry-run новый снимок не создаётся и старые снимки не удаляются.\n'
    exit 0
fi

rsync "${RSYNC_ARGS[@]}" "$SOURCE_DIR/" "$SNAPSHOT_DIR/"

CKPT_COUNT="$(find "$SNAPSHOT_DIR" -name '*.pt' -not -path '*/.git/*' 2>/dev/null | wc -l)"
SNAPSHOT_SIZE="$(du -sh "$SNAPSHOT_DIR" 2>/dev/null | cut -f1 || echo '?')"

cat > "$SNAPSHOT_DIR/BACKUP_METADATA.txt" <<EOF
AGI / PHEF backup metadata
Created: $(date --iso-8601=seconds)
Source: $SOURCE_DIR
Snapshot: $SNAPSHOT_DIR
Snapshot size: $SNAPSHOT_SIZE
Checkpoints (*.pt): $CKPT_COUNT
Mounted filesystem: $MOUNT_SOURCE at $MOUNT_POINT
Snapshot type: full independent copy; no hard links, symbolic links or link-dest are used
Included: source, docs, phef/results (weights), phef/logs, phef/output, git history
Excluded: .venv, mypy/pytest/ruff caches, Python bytecode, egg-info, coverage, reports, *.pid
Host: $(hostname)
Python: $(python3 --version 2>&1 || echo 'not available')
Git commit: $(git -C "$SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null || echo 'not available')
Git branch: $(git -C "$SOURCE_DIR" branch --show-current 2>/dev/null || echo 'not available')
Git remote: $(git -C "$SOURCE_DIR" remote get-url origin 2>/dev/null || echo 'not available')
Git status:
$(git -C "$SOURCE_DIR" status --short 2>/dev/null || true)
EOF

mapfile -t SNAPSHOTS < <(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
COUNT="${#SNAPSHOTS[@]}"
if (( COUNT > KEEP_SNAPSHOTS )); then
    REMOVE_COUNT=$((COUNT - KEEP_SNAPSHOTS))
    printf '\nУдаляются %s самых старых снимка(ов):\n' "$REMOVE_COUNT"
    for ((i = 0; i < REMOVE_COUNT; i++)); do
        OLD_SNAPSHOT="$BACKUP_ROOT/${SNAPSHOTS[$i]}"
        printf '  %s\n' "$OLD_SNAPSHOT"
        rm -rf --one-file-system "$OLD_SNAPSHOT"
    done
fi

printf '\nBackup успешно создан: %s (%s, чекпойнтов: %s)\n' "$SNAPSHOT_DIR" "$SNAPSHOT_SIZE" "$CKPT_COUNT"
printf 'Метаданные: %s/BACKUP_METADATA.txt\n' "$SNAPSHOT_DIR"
printf '\nТекущие снимки:\n'
find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | LC_ALL=C sort
